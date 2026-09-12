/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

document.addEventListener('DOMContentLoaded', () => {
    const infoDiv = document.querySelector('.info');
    const mainDiv = document.querySelector('.main');
    const reportSectionDiv = document.querySelector('.report-section');
    const viewDemoButton = document.getElementById('view-demo-button');
    const backToInfoButton = document.getElementById('back-to-info-button');
    const caseSelectorTabsContainer = document.getElementById('case-selector-tabs-container');
    const reportTextDisplay = document.getElementById('report-text-display');
    const explanationOutput = document.getElementById('explanation-output');
    const explanationContent = document.getElementById('explanation-content');
    const explanationError = document.getElementById('explanation-error');
    const imageContainer = document.getElementById('image-container');
    const reportImage = document.getElementById('report-image');
    const imageLoading = document.getElementById('image-loading');
    const imageError = document.getElementById('image-error');
    const imageModalityHeader = document.getElementById('image-modality-header'); // Get reference to the header
    const ctImageNote = document.getElementById('ct-image-note');
    const appLoading = document.getElementById('app-loading');
    const appError = document.getElementById('app-error');

    // Upload mode elements
    const uploadSection = document.getElementById('upload-section');
    const imageSection = document.querySelector('.image-section');
    const reportSection = document.querySelector('.report-section');
    const uploadFilesInput = document.getElementById('upload-files');
    const uploadModality = document.getElementById('upload-modality');
    const uploadQuestion = document.getElementById('upload-question');
    const uploadAnalyzeButton = document.getElementById('upload-analyze');
    const uploadStatus = document.getElementById('upload-status');
    const uploadError = document.getElementById('upload-error');
    const uploadPreviewGrid = document.getElementById('upload-preview-grid');
    const uploadResult = document.getElementById('upload-result');
    let currentMode = 'report';

    function renderMarkdown(el, text) {
        if (!el) return;
        if (window.marked && typeof window.marked.parse === 'function') {
            el.innerHTML = window.marked.parse(text || '', { breaks: true, gfm: true });
        } else {
            el.textContent = text || '';
        }
    }

    function setMode(mode) {
        currentMode = mode;
        const uploadTabButton = caseSelectorTabsContainer?.querySelector('[data-upload-tab]');
        if (mode === 'upload') {
            if (uploadSection) uploadSection.style.display = 'block';
            if (imageSection) imageSection.style.display = 'none';
            if (reportSection) reportSection.style.display = 'none';
            if (ctImageNote) ctImageNote.style.display = 'none';
            if (uploadTabButton) setActiveCaseButton(uploadTabButton);
        } else {
            if (uploadSection) uploadSection.style.display = 'none';
            if (imageSection) imageSection.style.display = '';
            if (reportSection) reportSection.style.display = '';
        }
    }

    let availableReports = [];
    let currentReportName = null;
    let currentReportDetails = null;

    let explainAbortController = null;
    let reportLoadAbortController = null;
    let appLoadingTimeout = null;
    let explanationLoadingTimer = null;

    function initialize() {
        try {
            const reportsDataElement = document.getElementById('reports-data');
            if (reportsDataElement) {
                availableReports = JSON.parse(reportsDataElement.textContent);
            } else {
                displayAppError("Failed to load report list.");
                return;
            }
        } catch (e) {
            displayAppError("Failed to parse report list.");
            return;
        }

        if (availableReports.length === 0) {
            displayAppError("No reports available.");
            return;
        }

        if (viewDemoButton && infoDiv && mainDiv) {
            viewDemoButton.addEventListener('click', () => {
                infoDiv.style.display = 'none';
                mainDiv.style.display = 'grid';
                if (currentReportName) {
                    loadReportDetails(currentReportName);
                }
            });
        }

        if (backToInfoButton && infoDiv && mainDiv) {
            backToInfoButton.addEventListener('click', () => {
                abortOngoingRequests();
                mainDiv.style.display = 'none';
                infoDiv.style.display = 'flex';
                clearAllOutputs();
                currentReportDetails = null;
                reportImage.src = '';
                document.title = "Radiology Report Explainer";
            });
        }

        if (caseSelectorTabsContainer) {
            caseSelectorTabsContainer.addEventListener('click', handleCaseSelectionClick);
        }

        // --- Upload mode wiring ---
        const uploadTabButton = caseSelectorTabsContainer?.querySelector('[data-upload-tab]');
        if (uploadTabButton && uploadSection) {
            uploadTabButton.addEventListener('click', () => {
                abortOngoingRequests();
                setMode('upload');
            });
        }
        if (uploadAnalyzeButton) {
            uploadAnalyzeButton.addEventListener('click', handleUploadAnalyze);
        }

        // --- IDC sample mode wiring ---
        document.querySelectorAll('.idc-button').forEach(btn => {
            btn.addEventListener('click', () => {
                const modality = btn.dataset.idcModality;
                if (modality) handleIdcSample(modality);
            });
        });

        const uploadFetchButton = document.getElementById('upload-fetch-samples');
        if (uploadFetchButton) {
            uploadFetchButton.addEventListener('click', handleFetchSamples);
        }

        // Clickable sentences in the generated explanation -> plain-language
        // wording for the clicked sentence (mirrors the report sentence flow).
        if (uploadResult) {
            uploadResult.addEventListener('click', handleRespSentenceClick);
        }

        reportTextDisplay.addEventListener('click', handleSentenceClick);

        const firstCaseButton = caseSelectorTabsContainer?.querySelector('.nav-button-case');
        if (firstCaseButton) {
            currentReportName = firstCaseButton.dataset.reportName;
            setActiveCaseButton(firstCaseButton);
            loadReportDetails(currentReportName);
        } else {
            displayAppError("No cases found to load initially.");
        }
    }

    function handleCaseSelectionClick(event) {
        const clickedButton = event.target.closest('.nav-button-case');
        if (!clickedButton) return;

        const selectedName = clickedButton.dataset.reportName;
        if (selectedName && selectedName !== currentReportName) {
            abortOngoingRequests();
            currentReportName = selectedName;
            setMode('report');
            setActiveCaseButton(clickedButton);
            loadReportDetails(currentReportName);
        }
    }

    async function handleSentenceClick(event) {
        const clickedElement = event.target;
        if (!clickedElement.classList.contains('report-sentence') || clickedElement.tagName !== 'SPAN') return;

        const sentenceText = clickedElement.dataset.sentence;
        if (!sentenceText || !currentReportName) return;

        abortOngoingRequests(['report']);
        explainAbortController = new AbortController();
        
        document.querySelectorAll('#report-text-display .selected-sentence').forEach(el => el.classList.remove('selected-sentence'));
        clickedElement.classList.add('selected-sentence');

        adjustExplanationPosition(clickedElement);

        try {
            await Promise.all([
                fetchExplanation(sentenceText, explainAbortController.signal),
            ]);
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error("Error during sentence processing:", error);
            }
        }
    }

    async function loadReportDetails(reportName) {
        abortOngoingRequests();
        reportLoadAbortController = new AbortController();
        const signal = reportLoadAbortController.signal;

        setLoadingState(true, 'report');
        clearAllOutputs(true);

        try {
            const response = await fetch(`/get_report_details/${encodeURIComponent(reportName)}`, { signal });
            if (signal.aborted) return;

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }

            currentReportDetails = await response.json();
            if (signal.aborted) return;

            document.title = `${reportName} - Radiology Explainer`;

            // Update the image modality header
            if (imageModalityHeader && currentReportDetails.image_type) {
                imageModalityHeader.textContent = currentReportDetails.image_type;
            }

            // Show/hide CT note based on image_type and image_file presence
            if (ctImageNote) {
                if (currentReportDetails.image_type === 'CT' && currentReportDetails.image_file) {
                    ctImageNote.style.display = 'block';
                } else {
                    ctImageNote.style.display = 'none';
                }
            }


            if (currentReportDetails.image_file) {
                const imageUrl = `${currentReportDetails.image_file}`;

                reportImage.onload = null;
                reportImage.onerror = null;

                reportImage.onload = () => {
                    imageLoading.style.display = 'none';
                    reportImage.style.display = 'block';
                    imageError.style.display = 'none';
                };
                reportImage.onerror = () => {
                    imageLoading.style.display = 'none';
                    reportImage.style.display = 'none';
                    displayImageError("Failed to load image file.");
                };

                reportImage.src = imageUrl;
                reportImage.alt = `Radiology Image for ${reportName}`;
            } else {
                displayImageError("Image path not configured for this report.");
                if (ctImageNote) ctImageNote.style.display = 'none'; // Ensure note is hidden if no image path
            }

            renderReportTextWithLineBreaks(currentReportDetails.text || '');
        } catch (error) {
            if (error.name !== 'AbortError') {
                displayReportTextError(`Failed to load report: ${error.message}`);
                // Clean up UI elements on report load error.
                if (reportImage) {
                    reportImage.style.display = 'none';
                    reportImage.src = '';
                    reportImage.onload = null;
                    reportImage.onerror = null;
                }
                if (imageLoading) imageLoading.style.display = 'none';
                if (imageError) imageError.style.display = 'none';
                if (ctImageNote) ctImageNote.style.display = 'none';
                clearExplanationAndLocationUI();
            }
        } finally {
            if (reportLoadAbortController?.signal === signal) {
                reportLoadAbortController = null;
            }
            if (!signal.aborted) {
                setLoadingState(false, 'report');
            }
        }
    }

    function renderReportTextWithLineBreaks(text) {
        reportTextDisplay.innerHTML = '';
        reportTextDisplay.classList.remove('loading', 'error');

        const lines = text.split('\n');
        if (lines.length === 0 || (lines.length === 1 && !lines[0].trim())) {
            reportTextDisplay.textContent = 'Report text is empty or could not be processed.';
            return;
        }

        lines.forEach((line, index) => {
            const trimmedLine = line.trim();
            if (trimmedLine !== '') {
                const sentences = splitSentences(trimmedLine);
                sentences.forEach(sentence => {
                    if (sentence) {
                        const span = document.createElement('span');
                        span.textContent = sentence + ' ';
                        if (!sentence.includes('Image source: ') && sentence.includes(' ') || !sentence.includes(':')) {
                            span.classList.add('report-sentence');
                        }
                        span.dataset.sentence = sentence;
                        reportTextDisplay.appendChild(span);
                    }
                });
            }
            if (index < lines.length - 1) {
                reportTextDisplay.appendChild(document.createElement('br'));
            }
        });
    }

    async function fetchExplanation(sentence, signal) {
        explanationError.style.display = 'none';
        if (!currentReportName) {
            displayExplanationError("No report selected.");
            return;
        }

    if (explanationLoadingTimer) {
        clearTimeout(explanationLoadingTimer);
    }

    explanationLoadingTimer = setTimeout(() => {
        if (!signal.aborted) { // Only add if not already aborted
            explanationOutput.classList.add('loading');
            explanationContent.textContent = '';
        }
        explanationLoadingTimer = null; // Timer has done its job or been cleared
    }, 150); // 150ms delay, adjust as needed

        try {
            const response = await fetch('/explain', {
                method: 'POST',
            headers: { 'Content-Type': 'application/json' }, // prettier-ignore
                body: JSON.stringify({ sentence, report_name: currentReportName }),
                signal
            });

            if (signal.aborted) return;

            if (!response.ok) {
            if (explanationLoadingTimer) clearTimeout(explanationLoadingTimer);
            explanationLoadingTimer = null;
            explanationOutput.classList.remove('loading');
                const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }

            const data = await response.json();
            if (signal.aborted) return;

        if (explanationLoadingTimer) clearTimeout(explanationLoadingTimer);
        explanationLoadingTimer = null;
            explanationOutput.classList.remove('loading');
            requestAnimationFrame(() => {
                renderMarkdown(explanationContent, data.explanation || "No explanation content received.");
                adjustExplanationPosition();
            });
        } catch (error) {
        if (explanationLoadingTimer) clearTimeout(explanationLoadingTimer);
        explanationLoadingTimer = null;
            explanationOutput.classList.remove('loading');
            if (error.name !== 'AbortError') {
                displayExplanationError(`Explanation Error: ${error.message}`);
            }
        }
    }

    async function handleUploadAnalyze() {
        if (!uploadFilesInput || !uploadFilesInput.files || uploadFilesInput.files.length === 0) {
            displayUploadError('Please select at least one file to upload.');
            return;
        }

        abortOngoingRequests();
        if (uploadError) uploadError.style.display = 'none';
        if (uploadResult) uploadResult.innerHTML = '';
        if (uploadPreviewGrid) uploadPreviewGrid.innerHTML = '';
        if (uploadStatus) {
            uploadStatus.textContent = 'Analyzing... The model endpoint may need to warm up for a few minutes if it was idle.';
            uploadStatus.style.display = 'block';
        }
        if (uploadAnalyzeButton) uploadAnalyzeButton.disabled = true;

        const formData = new FormData();
        for (let i = 0; i < uploadFilesInput.files.length; i++) {
            formData.append('files', uploadFilesInput.files[i]);
        }
        if (uploadModality && uploadModality.value) {
            formData.append('modality', uploadModality.value);
        }
        if (uploadQuestion && uploadQuestion.value.trim()) {
            formData.append('question', uploadQuestion.value.trim());
        }

        try {
            const response = await fetch('/upload_explain', { method: 'POST', body: formData });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.error || `HTTP error ${response.status}`);
            }

            if (uploadStatus) uploadStatus.style.display = 'none';

            const modalityLabels = { 'X-ray': 'Chest X-Ray', 'CT': 'CT', 'MRI': 'MRI' };
            const modalityLabel = modalityLabels[data.modality] || data.modality || 'Image';
            const meta = data.total_slices > 1
                ? `${modalityLabel} · ${data.total_slices} slices in series (using ${data.prompt_slices})`
                : `${modalityLabel} · single image`;

            if (uploadResult) {
                renderMarkdown(uploadResult, `${meta}\n\n${data.explanation || ''}`);
                makeResponseSentences(uploadResult);
                hideRespExplain();
            }
            if (data.previews && data.previews.length && uploadPreviewGrid) {
                renderPreviewGrid(data.previews);
            }
        } catch (error) {
            if (uploadStatus) uploadStatus.style.display = 'none';
            displayUploadError(`Upload Error: ${error.message}`);
        } finally {
            if (uploadAnalyzeButton) uploadAnalyzeButton.disabled = false;
        }
    }

    async function handleIdcSample(modality) {
        abortOngoingRequests();
        if (uploadError) uploadError.style.display = 'none';
        if (uploadResult) uploadResult.innerHTML = '';
        if (uploadPreviewGrid) uploadPreviewGrid.innerHTML = '';

        const idcButtons = document.querySelectorAll('.idc-button');
        idcButtons.forEach(b => { b.disabled = true; });

        const poolCounts = currentPoolCounts;
        if (uploadStatus) {
            uploadStatus.textContent =
                (poolCounts && poolCounts[modality])
                    ? `Analyzing a random ${modality} sample from the pre-fetched pool...`
                    : `Fetching a public ${modality} cancer sample from IDC... This can take a minute.`;
            uploadStatus.style.display = 'block';
        }

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 300000);

        try {
            const data = await (async () => {
                const response = await fetch('/idc_explain', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ modality, question: (uploadQuestion && uploadQuestion.value.trim()) || '' }),
                    signal: controller.signal
                });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.error || `HTTP error ${response.status}`);
                }
                return payload;
            })();

            if (uploadStatus) uploadStatus.style.display = 'none';
            currentPoolCounts = data.pool_counts || currentPoolCounts;

            const modalityLabels = { 'X-ray': 'Chest X-Ray', 'CT': 'CT', 'MRI': 'MRI' };
            const modalityLabel = modalityLabels[data.modality] || data.modality || 'Image';
            const meta = data.total_slices > 1
                ? `${modalityLabel} · ${data.total_slices} slices in series (using ${data.prompt_slices})`
                : `${modalityLabel} · single image`;
            const sourceLine = [data.source, data.body_part]
                .filter(Boolean)
                .join(' · ');
            const header = `**${meta}**` + (sourceLine ? `\n*Source: ${sourceLine}*` : '');

            if (uploadResult) {
                renderMarkdown(uploadResult, `${header}\n\n${data.explanation || ''}`);
                makeResponseSentences(uploadResult);
                hideRespExplain();
            }
            if (data.previews && data.previews.length && uploadPreviewGrid) {
                renderPreviewGrid(data.previews);
            }
        } catch (error) {
            if (uploadStatus) uploadStatus.style.display = 'none';
            const message = (error.name === 'AbortError')
                ? 'The IDC sample request timed out. Please try again.'
                : error.message;
            displayUploadError(`IDC Sample Error: ${message}`);
        } finally {
            clearTimeout(timeout);
            idcButtons.forEach(b => { b.disabled = false; });
        }
    }

    let currentPoolCounts = null;
    let fetchSamplesController = null;

    async function handleFetchSamples() {
        abortOngoingRequests();
        abortOngoingFetchSamples();
        if (uploadError) uploadError.style.display = 'none';
        if (uploadResult) uploadResult.innerHTML = '';
        if (uploadPreviewGrid) uploadPreviewGrid.innerHTML = '';
        hideRespExplain();

        const idcButtons = document.querySelectorAll('.idc-button');
        const fetchButton = document.getElementById('upload-fetch-samples');
        idcButtons.forEach(b => { b.disabled = true; });
        if (fetchButton) fetchButton.disabled = true;

        if (uploadStatus) {
            uploadStatus.textContent =
                'Fetching 10 public IDC samples (3 X-ray, 4 CT, 3 MRI)... This can take a few minutes.';
            uploadStatus.style.display = 'block';
        }

        fetchSamplesController = new AbortController();
        const timeout = setTimeout(() => {
            if (fetchSamplesController) fetchSamplesController.abort();
        }, 600000);

        try {
            const response = await fetch('/idc_fetch_samples', {
                method: 'POST',
                signal: fetchSamplesController.signal
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.error || `HTTP error ${response.status}`);
            }

            currentPoolCounts = data.by_modality || {};
            if (uploadStatus) uploadStatus.style.display = 'none';
            const counts = (data.by_modality || {});
            const parts = ['X-ray', 'CT', 'MRI']
                .map(m => `${m}: ${counts[m] ?? 0}`)
                .join(' · ');
            const message = `Staged **${data.samples ?? 0}** IDC samples.\n` +
                `_${parts}_\n\n` +
                `Now click **X-Ray**, **CT** or **MRI** to analyze a random sample instantly ` +
                `(click any response sentence to see it explained in plainer terms).`;
            if (uploadResult) {
                renderMarkdown(uploadResult, message);
                makeResponseSentences(uploadResult);
            }
        } catch (error) {
            if (uploadStatus) uploadStatus.style.display = 'none';
            const message = (error.name === 'AbortError')
                ? 'Fetching samples timed out after 10 minutes. Try again.'
                : error.message;
            displayUploadError(`Fetch Samples Error: ${message}`);
        } finally {
            clearTimeout(timeout);
            fetchSamplesController = null;
            idcButtons.forEach(b => { b.disabled = false; });
            if (fetchButton) fetchButton.disabled = false;
        }
    }

    function abortOngoingFetchSamples() {
        if (fetchSamplesController) {
            fetchSamplesController.abort();
            fetchSamplesController = null;
        }
    }

    function splitResponseSentences(text) {
        return (text || '').match(/[^.!?\n]+[.!?]*(?:\s+|$)/g) || [text];
    }

    function makeResponseSentences(root) {
        if (!root) return;
        root.querySelectorAll('p, li').forEach(el => {
            const text = el.textContent.trim();
            if (text.length < 3) return;
            const sentences = splitResponseSentences(text);
            if (sentences.length <= 1 && !text.endsWith('.') && !text.endsWith('!')
                && !text.endsWith('?')) {
                return;
            }
            el.textContent = '';
            sentences.forEach(s => {
                const span = document.createElement('span');
                span.className = 'resp-sentence';
                span.textContent = s + ' ';
                span.dataset.sentence = s.trim();
                el.appendChild(span);
            });
        });
    }

    function hideRespExplain() {
        const box = document.getElementById('resp-explain-box');
        const content = document.getElementById('resp-explain-content');
        if (box) { box.style.display = 'none'; box.classList.remove('loading'); }
        if (content) content.innerHTML = '';
    }

    async function handleRespSentenceClick(event) {
        const span = event.target.closest('.resp-sentence');
        if (!span) return;
        const sentence = span.dataset.sentence;
        if (!sentence || !sentence.trim()) return;

        const box = document.getElementById('resp-explain-box');
        const content = document.getElementById('resp-explain-content');
        if (!box || !content) return;

        uploadResult.querySelectorAll('.resp-sentence').forEach(s => s.classList.remove('active'));
        span.classList.add('active');
        box.style.display = 'block';
        box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        content.innerHTML = 'Generating a simpler explanation...';
        box.classList.add('loading');

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 120000);
        try {
            const resp = await fetch('/explain_sentence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sentence,
                    modality: uploadModality && uploadModality.value
                        ? uploadModality.value
                        : 'Medical Image'
                }),
                signal: controller.signal
            });
            const data = await resp.json().catch(() => ({}));
            box.classList.remove('loading');
            if (!resp.ok) {
                throw new Error(data.error || `HTTP error ${resp.status}`);
            }
            renderMarkdown(content, data.explanation || '');
            makeResponseSentences(content);
        } catch (error) {
            box.classList.remove('loading');
            content.textContent = (error.name === 'AbortError')
                ? 'The explanation request timed out. Please try again.'
                : `Error: ${error.message}`;
        } finally {
            clearTimeout(timeout);
        }
    }

    function renderPreviewGrid(previews) {
        previews.forEach(prev => {
            const cell = document.createElement('div');
            cell.className = 'upload-preview-cell';
            const img = document.createElement('img');
            img.className = 'upload-preview-thumb';
            img.src = prev.data_url;
            img.alt = prev.label || 'slice';
            const label = document.createElement('div');
            label.className = 'upload-preview-label';
            label.textContent = prev.label || '';
            cell.appendChild(img);
            cell.appendChild(label);
            uploadPreviewGrid.appendChild(cell);
        });
    }

    function displayUploadError(message) {
        if (uploadError) {
            uploadError.textContent = message;
            uploadError.style.display = 'block';
        }
    }

    function setLoadingState(isLoading, type = 'all') {
        if (type === 'all' || type === 'report') {
            if (isLoading) {
                if (appLoadingTimeout) {
                    clearTimeout(appLoadingTimeout);
                    appLoadingTimeout = null;
                }
                // Cleanup immediate error/content states
                if (appError) appError.style.display = 'none';
                if (reportImage) reportImage.style.display = 'none';
                if (imageError) imageError.style.display = 'none';
                if (ctImageNote) ctImageNote.style.display = 'none';
                clearExplanationAndLocationUI();

                appLoadingTimeout = setTimeout(() => {
                    if (appLoading) appLoading.style.display = 'block';

                    // Show content-specific loaders only if timeout fires
                    if (reportTextDisplay) {
                        reportTextDisplay.innerHTML = 'Loading...';
                        reportTextDisplay.classList.add('loading');
                        reportTextDisplay.classList.remove('error');
                    }
                    if (imageLoading) imageLoading.style.display = 'block';
                }, 200); // Delay for app and content loading indicators (e.g., 200ms)

            } else {
                if (appLoadingTimeout) {
                    clearTimeout(appLoadingTimeout);
                    appLoadingTimeout = null;
                }
                if (appLoading) appLoading.style.display = 'none';
            }
        }
    }
    function clearAllOutputs(keepReportTextLoading = false) {
        if (!keepReportTextLoading && reportTextDisplay) {
            reportTextDisplay.innerHTML = 'Select a report to view its text.';
            reportTextDisplay.classList.remove('loading', 'error');
        }
        if (reportImage) {
            reportImage.style.display = 'none';
            reportImage.onload = null;
            reportImage.onerror = null;
            reportImage.src = '';
        }
        if (imageError) imageError.style.display = 'none';
        if (ctImageNote) ctImageNote.style.display = 'none';
        clearExplanationAndLocationUI();
        if (appError) appError.style.display = 'none';
        // Reset image modality header on clear
        if (imageModalityHeader) {
             imageModalityHeader.textContent = 'Medical Image'; // Reset to default
        }
    }

    function clearExplanationAndLocationUI() {
        if (explanationContent) {
            explanationContent.textContent = 'Click a sentence to see the explanation here.';
        }
        if (explanationLoadingTimer) { // Clear any pending explanation loading timer
            clearTimeout(explanationLoadingTimer);
            explanationLoadingTimer = null;
        }
        explanationOutput.classList.remove('loading');
        explanationError.style.display = 'none';
        document.querySelectorAll('#report-text-display .selected-sentence').forEach(el => el.classList.remove('selected-sentence'));
    }

    function displayReportTextError(message) {
        reportTextDisplay.innerHTML = `<span class="error-message">${message}</span>`;
        reportTextDisplay.classList.add('error');
        reportTextDisplay.classList.remove('loading');
    }

    function displayAppError(message) {
        appError.textContent = `Error: ${message}`;
        appError.style.display = 'block';
        appLoading.style.display = 'none';
    }

    function displayImageError(message) {
        imageError.textContent = message;
        imageError.style.display = 'block';
        imageLoading.style.display = 'none';
        reportImage.style.display = 'none';
        if (ctImageNote) ctImageNote.style.display = 'none';
    }

    function displayExplanationError(message) {
        explanationError.textContent = message;
        explanationError.style.display = 'block';
        explanationOutput.classList.remove('loading');
        if (explanationContent) explanationContent.textContent = '';
    }

    function setActiveCaseButton(activeButton) {
        if (!caseSelectorTabsContainer) return;
        caseSelectorTabsContainer.querySelectorAll('.nav-button-case').forEach(btn => btn.classList.remove('active'));
        if (activeButton) activeButton.classList.add('active');
    }

    function splitSentences(text) {
        if (!text) return [];
        try {
            if (typeof nlp !== 'function') {
                const basicSentences = text.match(/[^.?!]+[.?!]['"]?(\s+|$)/g);
                return basicSentences ? basicSentences.map(s => s.trim()).filter(s => s.length > 0) : [];
            }
            const doc = nlp(text);
            return doc.sentences().out('array').map(s => s.trim()).filter(s => s.length > 0);
        } catch (e) {
            const basicSentences = text.match(/[^.?!]+[.?!]['"]?(\s+|$)/g);
            return basicSentences ? basicSentences.map(s => s.trim()).filter(s => s.length > 0) : [];
        }
    }

    function adjustExplanationPosition(clickedSentenceElement) {
        const targetSentenceElement = clickedSentenceElement || document.querySelector('#report-text-display .selected-sentence');
        if (!targetSentenceElement) return;

        const explanationSection = explanationOutput.closest('.explanation-section');

        if (explanationOutput && explanationSection && reportSectionDiv) {
            const sentenceRect = targetSentenceElement.getBoundingClientRect();
            const explanationSectionRect = explanationSection.getBoundingClientRect();
            
            // Use actual offsetHeight, fallback if not rendered. Accurate after rAF.
            const explanationHeight = explanationOutput.offsetHeight || 200; 
            
            // Initial top: align with sentence, relative to explanationSection.
            let newTop = sentenceRect.top - explanationSectionRect.top;
            
            // Absolute bottom of explanation box if placed at newTop.
            const explanationBoxAbsoluteBottom = explanationSectionRect.top + newTop + explanationHeight + 15; // 15px margin
            
            const viewportHeight = window.innerHeight;
            const pageBottomOverflow = explanationBoxAbsoluteBottom - viewportHeight;
            
            if (pageBottomOverflow > 0) {
                // Adjust newTop upwards if overflowing viewport bottom.
                newTop -= pageBottomOverflow;
            }
            
            // Prevent top from being negative (relative to its container).
            newTop = Math.max(0, newTop);
            
            explanationOutput.style.top = `${newTop}px`;
        }
    }

    function abortOngoingRequests(excludeTypes = []) {
        if (!excludeTypes.includes('report') && reportLoadAbortController) {
            reportLoadAbortController.abort();
            reportLoadAbortController = null;
        }
        if (!excludeTypes.includes('explain') && explainAbortController) {
            explainAbortController.abort();
            explainAbortController = null;
        }
        abortOngoingFetchSamples();
    }

    initialize();
});


// Make sure this is within your existing DOMContentLoaded listener,
// or wrap it in one if demo.js doesn't have a global one.
document.addEventListener('DOMContentLoaded', () => {

    // ... (any existing JavaScript code in demo.js)

    // --- BEGIN: Immersive Info Dialog Logic ---
    const infoButton = document.getElementById('info-button');
    const immersiveDialogOverlay = document.getElementById('immersive-info-dialog');
    const dialogCloseButton = document.getElementById('dialog-close-button');

    if (infoButton && immersiveDialogOverlay && dialogCloseButton) {
        const openDialog = () => {
            immersiveDialogOverlay.style.display = 'flex'; // Use flex as per CSS
            // Timeout to allow display:flex to apply before triggering transition
            setTimeout(() => {
                immersiveDialogOverlay.classList.add('active');
            }, 10); // Small delay
            document.body.style.overflow = 'hidden'; // Prevent background scroll
        };

        const closeDialog = () => {
            immersiveDialogOverlay.classList.remove('active');
            // Wait for opacity transition to finish before setting display to none
            setTimeout(() => {
                immersiveDialogOverlay.style.display = 'none';
            }, 300); // Must match CSS transition duration
            document.body.style.overflow = ''; // Restore background scroll
        };

        infoButton.addEventListener('click', openDialog);
        dialogCloseButton.addEventListener('click', closeDialog);

        // Dismissible: Close when clicking on the overlay (backdrop)
        immersiveDialogOverlay.addEventListener('click', (event) => {
            if (event.target === immersiveDialogOverlay) {
                closeDialog();
            }
        });

        // Dismissible: Close with Escape key
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && immersiveDialogOverlay.classList.contains('active')) {
                closeDialog();
            }
        });
    } else {
        // Log errors if elements are not found, helps in debugging
        if (!infoButton) console.error('Dialog trigger button (#info-button) not found.');
        if (!immersiveDialogOverlay) console.error('Immersive dialog (#immersive-info-dialog) not found.');
        if (!dialogCloseButton) console.error('Dialog close button (#dialog-close-button) not found.');
    }
    // --- END: Immersive Info Dialog Logic ---

});
