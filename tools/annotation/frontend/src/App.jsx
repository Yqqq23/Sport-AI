import { useEffect, useMemo, useRef, useState } from 'react'
import {
  API_BASE_URL,
  deleteAnnotation,
  downloadExport,
  exportInternalDataset,
  fetchAnnotationProgress,
  fetchAnnotations,
  fetchBasketballClasses,
  fetchRecognitionSummary,
  fetchReviewItems,
  fetchVideos,
  pollJob,
  runYoloPreannotationAsync,
  saveAnnotation,
  updateAnnotation,
  uploadVideo,
} from './api/client'

const buildApiUrl = (path) => (API_BASE_URL ? `${API_BASE_URL}${path}` : path)

// Box outline colours, keyed by class NAME rather than index so appending a
// class upstream cannot silently repaint an existing one.
const CLASS_COLORS = {
  player: '#4ade80',
  ball: '#fbbf24',
  rim: '#f87171',
  jersey: '#60a5fa',
  referee: '#c084fc',
}
const FALLBACK_CLASS_COLOR = '#e5e7eb'
const classColor = (name) => CLASS_COLORS[name] || FALLBACK_CLASS_COLOR

export default function App() {
  const [videos, setVideos] = useState([])
  const [selectedVideoId, setSelectedVideoId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [reviewItems, setReviewItems] = useState([])
  const [reviewIndex, setReviewIndex] = useState(0)
  const [loadingReviewItems, setLoadingReviewItems] = useState(false)
  const [recognitionSummary, setRecognitionSummary] = useState(null)
  const [annotationProgress, setAnnotationProgress] = useState(null)
  const [savedAnnotations, setSavedAnnotations] = useState([])
  const [editorBBox, setEditorBBox] = useState(null)
  const [correctedNumber, setCorrectedNumber] = useState('')
  const [labelStatus, setLabelStatus] = useState('correct')
  const [savingAnnotation, setSavingAnnotation] = useState(false)
  const [runningPreannotation, setRunningPreannotation] = useState(false)
  const [exportingDataset, setExportingDataset] = useState(false)
  const [jobState, setJobState] = useState(null)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  // --- 5-class annotation state ---
  const [classScheme, setClassScheme] = useState(null)
  const [classId, setClassId] = useState(null)
  const [trackId, setTrackId] = useState('0')
  // annotation_id of the saved box being edited; null means "drawing a new box"
  const [editingAnnotationId, setEditingAnnotationId] = useState(null)
  const [mockWarning, setMockWarning] = useState('')
  const mountedRef = useRef(true)
  const imageFrameRef = useRef(null)
  const interactionRef = useRef(null)
  const currentItemRef = useRef(null)
  const editorBBoxRef = useRef(null)

  const describeError = (err) => err?.responseText || err?.message || String(err)

  const numberBearingClassId = classScheme?.number_bearing_class_id ?? null
  const isJerseyClass = classId !== null && classId === numberBearingClassId
  // Only player and jersey boxes identify a person, so only they need a track
  // id. It is what dataset_versioning groups on to keep one player's frames on
  // one side of the train/val split.
  const identityClassNames = ['player', 'jersey']
  const currentClassName =
    classScheme?.classes?.find((item) => item.id === classId)?.name ?? ''
  const needsTrackId = identityClassNames.includes(currentClassName)

  const defaultBBoxForItem = (item) => {
    const width = item?.image_width || 640
    const height = item?.image_height || 480
    return [
      Math.round(width * 0.35),
      Math.round(height * 0.2),
      Math.round(width * 0.65),
      Math.round(height * 0.5),
    ]
  }

  const loadVideos = async () => {
    try {
      const rows = await fetchVideos()
      setVideos(rows)
      setSelectedVideoId((current) => {
        if (current && rows.some((row) => String(row.video_id ?? row.id) === current)) {
          return current
        }
        return rows.length > 0 ? String(rows[0].video_id ?? rows[0].id) : ''
      })
    } catch (err) {
      setError(describeError(err))
    }
  }

  const loadSavedAnnotations = async (videoId) => {
    const rows = await fetchAnnotations(videoId)
    setSavedAnnotations(rows)
    return rows
  }

  const loadProgress = async (videoId) => {
    const progress = await fetchAnnotationProgress(videoId)
    setAnnotationProgress(progress)
    return progress
  }

  const loadSummary = async (videoId) => {
    try {
      const summary = await fetchRecognitionSummary(videoId)
      setRecognitionSummary(summary)
      return summary
    } catch {
      setRecognitionSummary(null)
      return null
    }
  }

  const loadReviewFrames = async (videoId) => {
    if (!videoId) return
    setLoadingReviewItems(true)
    setError('')
    setMessage('')
    try {
      const response = await fetchReviewItems(videoId)
      const items = Array.isArray(response?.items) ? response.items : []
      setReviewItems(items)
      setReviewIndex(0)
      setMockWarning(response?.mock_data_warning || '')
      setMessage(response?.message || (items.length > 0 ? 'Frames loaded for annotation.' : 'No frames available for annotation.'))
      await loadSavedAnnotations(videoId)
      await loadProgress(videoId)
      await loadSummary(videoId)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setLoadingReviewItems(false)
    }
  }

  useEffect(() => {
    loadVideos()
    // The class list comes from the backend, which reads
    // basketball-v2/train/basketball.yaml. If this fails the UI stays disabled
    // rather than falling back to a guessed list: annotating against the wrong
    // class ids produces a dataset that is silently, unfixably mislabelled.
    fetchBasketballClasses()
      .then((scheme) => {
        setClassScheme(scheme)
        setClassId((current) => (current === null ? scheme.classes?.[0]?.id ?? null : current))
      })
      .catch((err) => setError(`Cannot load the class scheme: ${describeError(err)}`))
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!selectedVideoId) {
      setReviewItems([])
      setSavedAnnotations([])
      setAnnotationProgress(null)
      setRecognitionSummary(null)
      setReviewIndex(0)
      setEditorBBox(null)
      return
    }
    loadSavedAnnotations(selectedVideoId).catch(() => {})
    loadProgress(selectedVideoId).catch(() => {})
    loadSummary(selectedVideoId).catch(() => {})
  }, [selectedVideoId])

  const currentItem = reviewItems[reviewIndex] ?? null
  // Every box already saved on this frame, of any class. The frame can hold
  // many: several players, a ball, a rim, referees.
  const savedBoxes = currentItem?.saved_boxes ?? []
  const editingBox = editingAnnotationId
    ? savedBoxes.find((box) => box.annotation_id === editingAnnotationId) || null
    : null
  const currentAnnotation = currentItem
    ? savedAnnotations.find(
        (row) => Number(row.video_id) === Number(currentItem.video_id) && Number(row.prediction_id) === Number(currentItem.prediction_id),
      ) || null
    : null

  useEffect(() => {
    currentItemRef.current = currentItem
  }, [currentItem])

  useEffect(() => {
    editorBBoxRef.current = editorBBox
  }, [editorBBox])

  // Moving to another frame always drops back to "drawing a new box", so an
  // edit started on one frame cannot silently continue on the next.
  useEffect(() => {
    setEditingAnnotationId(null)
  }, [reviewIndex, selectedVideoId])

  useEffect(() => {
    if (!currentItem) {
      setEditorBBox(null)
      setCorrectedNumber('')
      setLabelStatus('correct')
      setTrackId('0')
      return
    }
    if (editingBox) {
      setEditorBBox(editingBox.corrected_bbox)
      setCorrectedNumber(editingBox.corrected_number ?? '')
      setLabelStatus(editingBox.label_status ?? 'correct')
      setClassId(editingBox.class_id)
      setTrackId(String(editingBox.track_id ?? 0))
      return
    }
    setEditorBBox(currentItem.jersey_bbox ?? defaultBBoxForItem(currentItem))
    setCorrectedNumber(currentItem.predicted_number ?? '')
    setLabelStatus('correct')
    setTrackId(String(currentItem.track_id ?? 0))
  }, [currentItem, editingBox])

  useEffect(() => {
    const onMove = (event) => {
      const interaction = interactionRef.current
      const frameRect = imageFrameRef.current?.getBoundingClientRect()
      const item = currentItemRef.current
      if (!interaction || !frameRect || !item) return
      const imageWidth = item.image_width || 1
      const imageHeight = item.image_height || 1
      const dx = ((event.clientX - interaction.startX) / frameRect.width) * imageWidth
      const dy = ((event.clientY - interaction.startY) / frameRect.height) * imageHeight

      setEditorBBox(() => {
        const [x1, y1, x2, y2] = interaction.initialBBox
        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        if (interaction.mode === 'drag') {
          let nx1 = x1 + dx
          let ny1 = y1 + dy
          nx1 = Math.max(0, Math.min(nx1, imageWidth - boxWidth))
          ny1 = Math.max(0, Math.min(ny1, imageHeight - boxHeight))
          return [Math.round(nx1), Math.round(ny1), Math.round(nx1 + boxWidth), Math.round(ny1 + boxHeight)]
        }
        const nx2 = Math.max(x1 + 8, Math.min(imageWidth, x2 + dx))
        const ny2 = Math.max(y1 + 8, Math.min(imageHeight, y2 + dy))
        return [x1, y1, Math.round(nx2), Math.round(ny2)]
      })
    }

    const onUp = () => {
      interactionRef.current = null
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  const startBBoxInteraction = (mode, event) => {
    if (!editorBBoxRef.current) return
    event.preventDefault()
    event.stopPropagation()
    interactionRef.current = {
      mode,
      startX: event.clientX,
      startY: event.clientY,
      initialBBox: [...editorBBoxRef.current],
    }
  }

  const handleUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    setMessage('Uploading video...')
    try {
      await uploadVideo(file)
      await loadVideos()
      setMessage('Video uploaded and frames extracted.')
    } catch (err) {
      setError(describeError(err))
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const handleRunPreannotation = async () => {
    if (!selectedVideoId) return
    setRunningPreannotation(true)
    setError('')
    setMessage('YOLO pre-annotation queued...')
    try {
      const queued = await runYoloPreannotationAsync(selectedVideoId)
      const finalJob = await pollJob(queued.job_id, {
        interval: 1000,
        onUpdate: (job) => {
          if (!mountedRef.current) return
          setJobState(job)
        },
      })
      if (!finalJob) return
      setJobState(finalJob)
      if (finalJob.status === 'completed') {
        setMessage(finalJob.message || 'YOLO pre-annotation completed.')
        await loadReviewFrames(selectedVideoId)
      } else {
        setError(finalJob.error_message || finalJob.message || 'YOLO pre-annotation failed.')
      }
    } catch (err) {
      setError(describeError(err))
    } finally {
      setRunningPreannotation(false)
    }
  }

  const validateAnnotation = () => {
    if (!currentItem || !editorBBox) return 'No frame selected.'
    if (classId === null) return 'Class scheme not loaded yet.'
    const [x1, y1, x2, y2] = editorBBox
    if (x1 >= x2 || y1 >= y2) return 'BBox is invalid.'
    // The number belongs to the jersey class only. Every other class stores no
    // number at all, so requiring one would be nonsense.
    if (isJerseyClass && labelStatus !== 'unclear' && !correctedNumber.trim()) {
      return 'Enter the jersey number, or set label status to "unclear" if it is not readable.'
    }
    return ''
  }

  const buildAnnotationPayload = () => ({
    // A negative prediction_id tells the backend this is a hand-drawn box; it
    // then allocates a unique id so the frame can hold as many boxes as needed.
    prediction_id: editingBox ? editingBox.prediction_id : -(Number(currentItem.frame_id) + 1),
    video_id: currentItem.video_id,
    frame_id: currentItem.frame_id,
    track_id: needsTrackId ? Number(trackId) || 0 : 0,
    class_id: classId,
    image_path: currentItem.frame_path || '',
    image_url: currentItem.image_url || '',
    original_bbox: currentItem.jersey_bbox || defaultBBoxForItem(currentItem),
    corrected_bbox: editorBBox,
    original_number: currentItem.predicted_number || '',
    corrected_number: isJerseyClass && labelStatus !== 'unclear' ? correctedNumber : '',
    label_status: labelStatus,
    image_width: currentItem.image_width,
    image_height: currentItem.image_height,
  })

  const reloadAnnotationState = async () => {
    if (!selectedVideoId) return
    await loadSavedAnnotations(selectedVideoId)
    await loadProgress(selectedVideoId)
    // Re-fetch review items so saved_boxes reflects the box just written.
    const response = await fetchReviewItems(selectedVideoId)
    const items = Array.isArray(response?.items) ? response.items : []
    setReviewItems(items)
    setMockWarning(response?.mock_data_warning || '')
  }

  const handleSave = async () => {
    const validationError = validateAnnotation()
    if (validationError) {
      setError(validationError)
      return
    }
    setSavingAnnotation(true)
    setError('')
    try {
      if (editingBox) {
        await updateAnnotation(editingBox.annotation_id, {
          corrected_bbox: editorBBox,
          class_id: classId,
          corrected_number: isJerseyClass && labelStatus !== 'unclear' ? correctedNumber : '',
          label_status: labelStatus,
          image_width: currentItem.image_width,
          image_height: currentItem.image_height,
        })
        setMessage(`Box ${editingBox.annotation_id} updated.`)
      } else {
        await saveAnnotation(buildAnnotationPayload())
        setMessage(`New ${currentClassName} box saved.`)
      }
      setEditingAnnotationId(null)
      await reloadAnnotationState()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSavingAnnotation(false)
    }
  }

  const handleDelete = async (annotationId) => {
    const targetId = annotationId ?? editingBox?.annotation_id
    if (!targetId) return
    setSavingAnnotation(true)
    setError('')
    try {
      await deleteAnnotation(targetId)
      setMessage(`Box ${targetId} deleted.`)
      if (editingAnnotationId === targetId) setEditingAnnotationId(null)
      await reloadAnnotationState()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSavingAnnotation(false)
    }
  }

  const startNewBox = () => {
    setEditingAnnotationId(null)
    setCorrectedNumber('')
    setLabelStatus('correct')
    setEditorBBox(currentItem ? defaultBBoxForItem(currentItem) : null)
    setMessage('Drawing a new box. Pick its class, drag and resize, then save.')
  }

  const handleDownload = async () => {
    if (!selectedVideoId) return
    if (!annotationProgress?.annotated_count) {
      setError('Please save at least one corrected annotation before downloading.')
      return
    }
    setExportingDataset(true)
    setError('')
    try {
      const result = await exportInternalDataset(selectedVideoId)
      await downloadExport(result.export_id)
      window.open(buildApiUrl(`/exports/${result.export_id}/download?raw=1`), '_blank', 'noopener,noreferrer')
      setMessage('Corrected dataset download started.')
    } catch (err) {
      setError(describeError(err))
    } finally {
      setExportingDataset(false)
    }
  }

  const selectedVideo = videos.find((video) => String(video.video_id ?? video.id) === selectedVideoId)
  const imageUrl = currentItem?.image_url ? buildApiUrl(currentItem.image_url) : ''

  const boxToPercentStyle = (bbox) => {
    const width = currentItem?.image_width || 1
    const height = currentItem?.image_height || 1
    const [x1, y1, x2, y2] = bbox
    return {
      left: `${(x1 / width) * 100}%`,
      top: `${(y1 / height) * 100}%`,
      width: `${((x2 - x1) / width) * 100}%`,
      height: `${((y2 - y1) / height) * 100}%`,
    }
  }

  const jerseyBoxStyle = useMemo(() => {
    if (!currentItem?.jersey_bbox) return null
    const [x1, y1, x2, y2] = currentItem.jersey_bbox
    const width = currentItem.image_width || 1
    const height = currentItem.image_height || 1
    return {
      left: `${(x1 / width) * 100}%`,
      top: `${(y1 / height) * 100}%`,
      width: `${((x2 - x1) / width) * 100}%`,
      height: `${((y2 - y1) / height) * 100}%`,
    }
  }, [currentItem])

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">basketball-v2 Annotation Tool</p>
        <h1>jersey-active-learning</h1>
        <p className="subcopy">
          Upload video, extract frames, draw boxes for the {classScheme?.class_count ?? 5} basketball
          classes, then export a YOLO dataset split by video.
        </p>
        {classScheme ? (
          <p className="subcopy">
            Classes from <code>{classScheme.source_of_truth}</code>:{' '}
            {classScheme.classes.map((item) => (
              <span key={item.id} style={{ color: classColor(item.name), marginRight: '0.75rem' }}>
                {item.id}={item.name}
              </span>
            ))}
          </p>
        ) : (
          <p className="error-text">Class scheme not loaded - annotation is disabled.</p>
        )}
      </header>

      {mockWarning ? (
        <section
          className="card"
          style={{ border: '2px solid #f87171', background: 'rgba(248,113,113,0.12)' }}
        >
          <h2 style={{ color: '#f87171', marginTop: 0 }}>MOCK DATA IN VIEW</h2>
          <p>{mockWarning}</p>
        </section>
      ) : null}

      <section className="card upload-card">
        <label className="upload-label">
          <span>{uploading ? 'Uploading...' : 'Upload Video'}</span>
          <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} />
        </label>
      </section>

      <section className="card">
        <div className="section-header">
          <h2>Videos</h2>
          <button type="button" onClick={loadVideos}>Refresh Videos</button>
        </div>
        <div className="video-list">
          {videos.length === 0 ? <p>No videos uploaded yet.</p> : null}
          {videos.map((video) => {
            const videoId = String(video.video_id ?? video.id)
            return (
              <article key={videoId} className={`video-row ${videoId === selectedVideoId ? 'video-row--active' : ''}`} onClick={() => setSelectedVideoId(videoId)}>
                <div>
                  <strong>{video.original_filename ?? video.filename}</strong>
                  <p>video_id: {videoId}</p>
                  <p>frame_count: {video.frame_count ?? 'N/A'}</p>
                </div>
                <div className="video-actions">
                  <span className="status">{video.status}</span>
                  <button type="button" onClick={(event) => { event.stopPropagation(); setSelectedVideoId(videoId) }}>Select Video</button>
                </div>
              </article>
            )
          })}
        </div>
      </section>

      <section className="card">
        <div className="section-header">
          <div>
            <h2>Simple Annotation Mode</h2>
            {selectedVideo ? <p className="selected-video-line">Selected: {selectedVideo.original_filename ?? selectedVideo.filename}</p> : null}
            {annotationProgress ? <p className="annotation-progress-line">Annotation Progress: {annotationProgress.annotated_count} / {annotationProgress.total_candidates} annotated ({annotationProgress.progress_percent}%)</p> : null}
            {recognitionSummary ? <p className="selected-video-line">YOLO Summary: {recognitionSummary.prediction_count} pre-annotations, average confidence {recognitionSummary.average_confidence}</p> : null}
          </div>
          <div className="action-row">
            <button type="button" onClick={handleRunPreannotation} disabled={!selectedVideoId || runningPreannotation}>
              {runningPreannotation ? 'Running YOLO Pre-annotation...' : 'Run YOLO Pre-annotation'}
            </button>
            <button type="button" onClick={() => loadReviewFrames(selectedVideoId)} disabled={!selectedVideoId || loadingReviewItems}>
              {loadingReviewItems ? 'Loading Frames...' : 'Load Frames for Annotation'}
            </button>
            <button type="button" onClick={handleDownload} disabled={!selectedVideoId || exportingDataset}>
              {exportingDataset ? 'Preparing Download...' : 'Download Corrected Dataset'}
            </button>
          </div>
        </div>

        {jobState ? (
          <div className="job-card">
            <p><strong>Job Status:</strong> {jobState.status}</p>
            <p><strong>Progress:</strong> {jobState.progress ?? 0}%</p>
            <p><strong>Message:</strong> {jobState.message || 'Waiting...'}</p>
          </div>
        ) : null}
        {message ? <p className="success-text">{message}</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
        {!selectedVideoId ? <p>Select a video first.</p> : null}
        {!loadingReviewItems && selectedVideoId && reviewItems.length === 0 && !error ? <p>No frames loaded yet. Click "Load Frames for Annotation".</p> : null}

        {currentItem ? (
          <div className="candidate-layout">
            <div className="candidate-stage">
              <div className="image-frame" ref={imageFrameRef}>
                <img src={imageUrl} alt={`Frame ${currentItem.frame_id}`} />
                {/* The pre-annotation box, shown only when one exists and it is
                    not mock. A mock box is not evidence of anything and drawing
                    it invites the annotator to trust it. */}
                {jerseyBoxStyle && !currentItem.is_mock_prediction ? (
                  <div className="jersey-box" style={jerseyBoxStyle} />
                ) : null}

                {/* Every box already saved on this frame, coloured by class.
                    Click one to edit it. */}
                {savedBoxes
                  .filter((box) => box.annotation_id !== editingAnnotationId)
                  .map((box) => (
                    <div
                      key={box.annotation_id}
                      style={{
                        ...boxToPercentStyle(box.corrected_bbox),
                        position: 'absolute',
                        border: `2px solid ${classColor(box.class_name)}`,
                        cursor: 'pointer',
                      }}
                      onClick={() => setEditingAnnotationId(box.annotation_id)}
                      title={`Click to edit ${box.class_name} box ${box.annotation_id}`}
                    >
                      <span
                        className="bbox-label"
                        style={{ background: classColor(box.class_name), color: '#111' }}
                      >
                        {box.class_name}
                        {box.corrected_number ? ` #${box.corrected_number}` : ''}
                      </span>
                    </div>
                  ))}

                {editorBBox ? (
                  <div
                    className="editable-jersey-box"
                    style={{
                      ...boxToPercentStyle(editorBBox),
                      borderColor: classColor(currentClassName),
                    }}
                    onMouseDown={(event) => startBBoxInteraction('drag', event)}
                  >
                    <span className="bbox-label">
                      {editingBox ? `Editing ${currentClassName}` : `New ${currentClassName}`}
                    </span>
                    <button type="button" className="bbox-resize-handle" onMouseDown={(event) => startBBoxInteraction('resize', event)} aria-label="Resize bbox" />
                  </div>
                ) : null}
              </div>
            </div>

            <div className="candidate-meta">
              <p><strong>Frame:</strong> {currentItem.frame_id}</p>
              <p><strong>Item:</strong> {reviewIndex + 1} / {reviewItems.length}</p>
              <p>
                <strong>Pre-annotation:</strong>{' '}
                {currentItem.is_mock_prediction ? (
                  <span style={{ color: '#f87171', fontWeight: 700 }}>
                    MOCK ({currentItem.prediction_source}) - invented, not detected
                  </span>
                ) : (
                  currentItem.prediction_source || 'manual'
                )}
              </p>
              {currentItem.prediction_id >= 0 && !currentItem.is_mock_prediction ? (
                <p>
                  <strong>Predicted number:</strong> {currentItem.predicted_number || '(empty)'}{' '}
                  <strong>conf:</strong> {currentItem.confidence}
                </p>
              ) : null}

              <p><strong>Boxes on this frame:</strong> {savedBoxes.length}</p>
              <ul className="saved-box-list">
                {savedBoxes.map((box) => (
                  <li key={box.annotation_id}>
                    <button
                      type="button"
                      onClick={() => setEditingAnnotationId(box.annotation_id)}
                      style={{
                        borderLeft: `6px solid ${classColor(box.class_name)}`,
                        fontWeight: box.annotation_id === editingAnnotationId ? 700 : 400,
                      }}
                    >
                      {box.class_name}
                      {box.corrected_number ? ` #${box.corrected_number}` : ''}
                      {box.track_id ? ` (track ${box.track_id})` : ''}
                      {box.label_status !== 'correct' ? ` [${box.label_status}]` : ''}
                    </button>
                    <button type="button" onClick={() => handleDelete(box.annotation_id)} disabled={savingAnnotation}>
                      delete
                    </button>
                  </li>
                ))}
              </ul>

              <button type="button" onClick={startNewBox} disabled={!classScheme}>
                + Add another box
              </button>

              <hr />
              <p>
                <strong>{editingBox ? `Editing box ${editingBox.annotation_id}` : 'New box'}</strong>
              </p>

              <label className="annotation-field">
                <span>Class</span>
                <select
                  value={classId ?? ''}
                  onChange={(event) => setClassId(Number(event.target.value))}
                  disabled={!classScheme}
                >
                  {(classScheme?.classes ?? []).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.id} - {item.name}
                    </option>
                  ))}
                </select>
              </label>

              {needsTrackId ? (
                <label className="annotation-field">
                  <span>Track id (same player = same id)</span>
                  <input
                    type="number"
                    min="0"
                    value={trackId}
                    onChange={(event) => setTrackId(event.target.value)}
                  />
                  <small>
                    0 means "unassigned". Only needed for split_strategy=track; splitting by
                    video works without it.
                  </small>
                </label>
              ) : null}

              {/* The number is NOT a class and never enters a YOLO label file.
                  It is exported to jersey_numbers.json as OCR ground truth. */}
              {isJerseyClass ? (
                <label className="annotation-field">
                  <span>Jersey number (0-99, OCR ground truth)</span>
                  <input
                    type="text"
                    value={correctedNumber}
                    onChange={(event) => setCorrectedNumber(event.target.value)}
                    disabled={labelStatus === 'unclear'}
                  />
                  <small>
                    Normalized by {classScheme?.jersey_number_rule}. Not a detection class -
                    exported separately to jersey_numbers.json.
                  </small>
                </label>
              ) : (
                <p>
                  <small>
                    No number field: only the{' '}
                    {classScheme?.number_bearing_class_name ?? 'jersey'} class carries one.
                  </small>
                </p>
              )}

              <label className="annotation-field">
                <span>Label status</span>
                <select value={labelStatus} onChange={(event) => setLabelStatus(event.target.value)}>
                  <option value="correct">correct</option>
                  <option value="wrong_bbox">wrong_bbox</option>
                  <option value="wrong_number">wrong_number</option>
                  <option value="unclear">unclear (excluded from export)</option>
                </select>
              </label>

              <button type="button" onClick={handleSave} disabled={savingAnnotation || !editorBBox || !classScheme}>
                {savingAnnotation ? 'Saving...' : editingBox ? 'Update Box' : 'Save New Box'}
              </button>
              {editingBox ? (
                <button type="button" onClick={() => setEditingAnnotationId(null)} disabled={savingAnnotation}>
                  Cancel edit
                </button>
              ) : null}

              <div className="nav-row">
                <button type="button" onClick={() => setReviewIndex((idx) => Math.max(0, idx - 1))} disabled={reviewIndex === 0}>Previous</button>
                <button type="button" onClick={() => setReviewIndex((idx) => Math.min(reviewItems.length - 1, idx + 1))} disabled={reviewIndex >= reviewItems.length - 1}>Next</button>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  )
}
