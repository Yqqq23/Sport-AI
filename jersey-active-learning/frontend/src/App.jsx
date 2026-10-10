import { useEffect, useMemo, useRef, useState } from 'react'
import {
  API_BASE_URL,
  deleteAnnotation,
  downloadExport,
  exportInternalDataset,
  fetchAnnotationProgress,
  fetchAnnotations,
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

export default function App() {
  const [videos, setVideos] = useState([])
  const [selectedVideoId, setSelectedVideoId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [reviewItems, setReviewItems] = useState([])
  const [reviewIndex, setReviewIndex] = useState(0)
  const [selectedBoxKey, setSelectedBoxKey] = useState('')
  const [draftBox, setDraftBox] = useState(null)
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
  const mountedRef = useRef(true)
  const imageFrameRef = useRef(null)
  const interactionRef = useRef(null)
  const currentItemRef = useRef(null)
  const editorBBoxRef = useRef(null)

  const describeError = (err) => err?.responseText || err?.message || String(err)

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
      setSelectedBoxKey('')
      setDraftBox(null)
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
      setSelectedBoxKey('')
      setDraftBox(null)
      setEditorBBox(null)
      return
    }
    loadSavedAnnotations(selectedVideoId).catch(() => {})
    loadProgress(selectedVideoId).catch(() => {})
    loadSummary(selectedVideoId).catch(() => {})
  }, [selectedVideoId])

  const reviewFrames = useMemo(() => {
    const frames = []
    const byFrame = new Map()
    reviewItems.forEach((item) => {
      const key = `${item.video_id}:${item.frame_id}`
      if (!byFrame.has(key)) {
        const frame = { ...item, items: [] }
        byFrame.set(key, frame)
        frames.push(frame)
      }
      byFrame.get(key).items.push(item)
    })
    return frames
  }, [reviewItems])

  const currentFrame = reviewFrames[reviewIndex] ?? null
  const currentFrameBoxes = useMemo(() => {
    if (!currentFrame) return []
    const predictionIds = new Set(currentFrame.items.map((item) => Number(item.prediction_id)))
    const boxes = currentFrame.items.map((item) => {
      const annotation = savedAnnotations.find(
        (row) => Number(row.video_id) === Number(item.video_id) && Number(row.prediction_id) === Number(item.prediction_id),
      ) || null
      return {
        key: `prediction:${item.prediction_id}`,
        item,
        annotation,
        bbox: annotation?.corrected_bbox ?? item.jersey_bbox ?? defaultBBoxForItem(item),
      }
    })
    savedAnnotations
      .filter(
        (row) => Number(row.frame_id) === Number(currentFrame.frame_id) && !predictionIds.has(Number(row.prediction_id)),
      )
      .forEach((annotation) => {
        boxes.push({
          key: `annotation:${annotation.annotation_id}`,
          annotation,
          bbox: annotation.corrected_bbox,
          item: {
            ...currentFrame,
            review_id: `annotation:${annotation.annotation_id}`,
            prediction_id: annotation.prediction_id,
            track_id: annotation.track_id,
            jersey_bbox: annotation.original_bbox,
            predicted_number: annotation.original_number,
            confidence: 0,
          },
        })
      })
    if (draftBox) boxes.push(draftBox)
    return boxes
  }, [currentFrame, draftBox, savedAnnotations])

  const selectedBox = currentFrameBoxes.find((box) => box.key === selectedBoxKey) ?? currentFrameBoxes[0] ?? null
  const currentItem = selectedBox?.item ?? null
  const currentAnnotation = selectedBox?.annotation ?? null

  useEffect(() => {
    if (!currentFrameBoxes.some((box) => box.key === selectedBoxKey)) {
      setSelectedBoxKey(currentFrameBoxes[0]?.key ?? '')
    }
  }, [currentFrameBoxes, selectedBoxKey])

  useEffect(() => {
    currentItemRef.current = currentItem
  }, [currentItem])

  useEffect(() => {
    editorBBoxRef.current = editorBBox
  }, [editorBBox])

  useEffect(() => {
    if (!currentItem) {
      setEditorBBox(null)
      setCorrectedNumber('')
      setLabelStatus('correct')
      return
    }
    setEditorBBox(selectedBox?.bbox ?? currentItem.jersey_bbox ?? defaultBBoxForItem(currentItem))
    setCorrectedNumber(currentAnnotation?.corrected_number ?? currentItem.predicted_number ?? '')
    setLabelStatus(currentAnnotation?.label_status ?? 'correct')
  }, [selectedBox?.key, selectedBox?.bbox, currentAnnotation?.annotation_id, currentAnnotation?.corrected_number, currentAnnotation?.label_status])

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
    const [x1, y1, x2, y2] = editorBBox
    if (x1 >= x2 || y1 >= y2) return 'BBox is invalid.'
    if (labelStatus !== 'unclear' && !correctedNumber.trim()) return 'Please enter jersey number or choose unclear.'
    return ''
  }

  const buildAnnotationPayload = () => ({
    prediction_id: selectedBoxKey === 'draft' ? null : currentItem.prediction_id,
    video_id: currentItem.video_id,
    frame_id: currentItem.frame_id,
    track_id: currentItem.track_id,
    image_path: currentItem.frame_path || '',
    image_url: currentItem.image_url || '',
    original_bbox: currentItem.jersey_bbox || defaultBBoxForItem(currentItem),
    corrected_bbox: editorBBox,
    original_number: currentItem.predicted_number || '',
    corrected_number: labelStatus === 'unclear' ? '' : correctedNumber,
    label_status: labelStatus,
    image_width: currentItem.image_width,
    image_height: currentItem.image_height,
  })

  const reloadAnnotationState = async () => {
    if (!selectedVideoId) return
    await loadSavedAnnotations(selectedVideoId)
    await loadProgress(selectedVideoId)
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
      let newAnnotationId = null
      if (currentAnnotation) {
        await updateAnnotation(currentAnnotation.annotation_id, {
          corrected_bbox: editorBBox,
          corrected_number: labelStatus === 'unclear' ? '' : correctedNumber,
          label_status: labelStatus,
          image_width: currentItem.image_width,
          image_height: currentItem.image_height,
        })
        setMessage('Annotation updated.')
      } else {
        const result = await saveAnnotation(buildAnnotationPayload())
        if (selectedBoxKey === 'draft') {
          newAnnotationId = result.annotation_id
        }
        setMessage('Annotation saved.')
      }
      await reloadAnnotationState()
      if (newAnnotationId) {
        setDraftBox(null)
        setSelectedBoxKey(`annotation:${newAnnotationId}`)
      }
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSavingAnnotation(false)
    }
  }

  const handleAddBox = () => {
    if (!currentFrame) return
    const item = {
      ...currentFrame,
      review_id: 'draft',
      prediction_id: null,
      track_id: 0,
      jersey_bbox: defaultBBoxForItem(currentFrame),
      predicted_number: '',
      confidence: 0,
    }
    setDraftBox({ key: 'draft', item, annotation: null, bbox: item.jersey_bbox })
    setSelectedBoxKey('draft')
    setError('')
    setMessage('New box added. Move or resize it, enter a jersey number, then save.')
  }

  const handleDelete = async () => {
    if (!currentItem) return
    if (selectedBoxKey === 'draft') {
      setDraftBox(null)
      setMessage('Unsaved box removed.')
      return
    }
    setSavingAnnotation(true)
    setError('')
    try {
      let annotationId = currentAnnotation?.annotation_id
      if (!annotationId) {
        const result = await saveAnnotation({
          ...buildAnnotationPayload(),
          corrected_number: '',
          label_status: 'unclear',
        })
        annotationId = result.annotation_id
      }
      await deleteAnnotation(annotationId)
      setDraftBox(null)
      setSelectedBoxKey('')
      const frameId = currentItem.frame_id
      const response = await fetchReviewItems(selectedVideoId)
      const items = Array.isArray(response?.items) ? response.items : []
      setReviewItems(items)
      const frameIds = [...new Set(items.map((item) => Number(item.frame_id)))]
      const nextIndex = frameIds.indexOf(Number(frameId))
      setReviewIndex(nextIndex >= 0 ? nextIndex : Math.min(reviewIndex, Math.max(frameIds.length - 1, 0)))
      setMessage('Box deleted.')
      await reloadAnnotationState()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSavingAnnotation(false)
    }
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

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">RA Annotation Tool</p>
        <h1>jersey-active-learning</h1>
        <p className="subcopy">Upload video, run YOLO pre-annotation, manually correct bbox and jersey number, then download the corrected dataset.</p>
      </header>

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
                {currentFrameBoxes.filter((box) => box.key !== selectedBox?.key).map((box, index) => {
                  const [x1, y1, x2, y2] = box.bbox
                  const width = box.item.image_width || 1
                  const height = box.item.image_height || 1
                  return (
                    <button
                      type="button"
                      key={box.key}
                      className="frame-jersey-box"
                      style={{
                        left: `${(x1 / width) * 100}%`,
                        top: `${(y1 / height) * 100}%`,
                        width: `${((x2 - x1) / width) * 100}%`,
                        height: `${((y2 - y1) / height) * 100}%`,
                      }}
                      onClick={() => setSelectedBoxKey(box.key)}
                      aria-label={`Select box ${index + 1}`}
                    >
                      <span>{box.annotation?.corrected_number || box.item.predicted_number || `Box ${index + 1}`}</span>
                    </button>
                  )
                })}
                {editorBBox ? (
                  <div
                    className="editable-jersey-box"
                    style={{
                      left: `${(editorBBox[0] / (currentItem.image_width || 1)) * 100}%`,
                      top: `${(editorBBox[1] / (currentItem.image_height || 1)) * 100}%`,
                      width: `${((editorBBox[2] - editorBBox[0]) / (currentItem.image_width || 1)) * 100}%`,
                      height: `${((editorBBox[3] - editorBBox[1]) / (currentItem.image_height || 1)) * 100}%`,
                    }}
                    onMouseDown={(event) => startBBoxInteraction('drag', event)}
                  >
                    <span className="bbox-label">{correctedNumber || currentItem.predicted_number || 'Edit'}</span>
                    <button type="button" className="bbox-resize-handle" onMouseDown={(event) => startBBoxInteraction('resize', event)} aria-label="Resize bbox" />
                  </div>
                ) : null}
              </div>
            </div>

            <div className="candidate-meta">
              <p><strong>Frame:</strong> {currentItem.frame_id}</p>
              <p><strong>Track:</strong> {currentItem.track_id}</p>
              <p><strong>Predicted Number:</strong> {currentItem.predicted_number || '(empty)'}</p>
              <p><strong>Confidence:</strong> {currentItem.confidence}</p>
              <p><strong>Source:</strong> {!currentItem.prediction_id || currentItem.prediction_id < 0 ? 'Manual frame annotation' : 'YOLO pre-annotation'}</p>
              <p><strong>Frame:</strong> {reviewIndex + 1} / {reviewFrames.length}, <strong>Boxes:</strong> {currentFrameBoxes.length}</p>

              <div className="box-selector" aria-label="Boxes in current frame">
                {currentFrameBoxes.map((box, index) => (
                  <button
                    type="button"
                    key={box.key}
                    className={box.key === selectedBox?.key ? 'box-selector__active' : ''}
                    onClick={() => setSelectedBoxKey(box.key)}
                  >
                    Box {index + 1}: {box.annotation?.corrected_number || box.item.predicted_number || 'new'}
                  </button>
                ))}
              </div>

              <label className="annotation-field">
                <span>Corrected jersey number</span>
                <input type="text" value={correctedNumber} onChange={(event) => setCorrectedNumber(event.target.value)} disabled={labelStatus === 'unclear'} />
              </label>

              <label className="annotation-field">
                <span>Label status</span>
                <select value={labelStatus} onChange={(event) => setLabelStatus(event.target.value)}>
                  <option value="correct">correct</option>
                  <option value="wrong_bbox">wrong_bbox</option>
                  <option value="wrong_number">wrong_number</option>
                  <option value="unclear">unclear</option>
                </select>
              </label>

              <button type="button" onClick={handleSave} disabled={savingAnnotation || !editorBBox}>
                {savingAnnotation ? 'Saving...' : currentAnnotation ? 'Update Annotation' : 'Save Annotation'}
              </button>
              <button type="button" onClick={handleAddBox} disabled={savingAnnotation || Boolean(draftBox)}>Add Box</button>
              <button type="button" className="danger-button" onClick={handleDelete} disabled={savingAnnotation}>{savingAnnotation ? 'Deleting...' : 'Delete Box'}</button>

              <div className="nav-row">
                <button type="button" onClick={() => { setDraftBox(null); setSelectedBoxKey(''); setReviewIndex((idx) => Math.max(0, idx - 1)) }} disabled={reviewIndex === 0}>Previous Frame</button>
                <button type="button" onClick={() => { setDraftBox(null); setSelectedBoxKey(''); setReviewIndex((idx) => Math.min(reviewFrames.length - 1, idx + 1)) }} disabled={reviewIndex >= reviewFrames.length - 1}>Next Frame</button>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  )
}
