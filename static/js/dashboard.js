const HISTORY_KEYS = {
    decisionTree: 'ensembleStudio.decisionTreeHistory',
    randomForest: 'ensembleStudio.randomForestHistory',
};

let bootstrapAnimationId = null;
let currentBootstrapData = null;
let currentDecisionTreeData = null;
let currentRandomForestData = null;

function initHomeScene() {
    const canvas = document.getElementById('home-scene');
    if (!canvas || typeof THREE === 'undefined') {
        return;
    }
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 18;

    const geometry = new THREE.BufferGeometry();
    const particleCount = 1400;
    const positions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particleCount * 3; index += 3) {
        positions[index] = (Math.random() - 0.5) * 28;
        positions[index + 1] = (Math.random() - 0.5) * 18;
        positions[index + 2] = (Math.random() - 0.5) * 24;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ color: 0x60a5fa, size: 0.06, transparent: true, opacity: 0.85 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const resize = () => {
        renderer.setSize(window.innerWidth, window.innerHeight, false);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener('resize', resize);

    const animate = () => {
        points.rotation.y += 0.0009;
        points.rotation.x += 0.00035;
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
    };
    animate();
}

function bindHomePage() {
    initHomeScene();
    const input = document.getElementById('dataset-upload-input');
    const status = document.getElementById('upload-status');
    const dropzone = document.getElementById('dataset-dropzone');
    const progressWrap = document.getElementById('upload-progress-wrap');
    const progressBar = document.getElementById('upload-progress-bar');

    const uploadFile = (file) => {
        const formData = new FormData();
        formData.append('file', file);
        if (status) {
            status.textContent = 'Uploading dataset and profiling columns...';
        }
        if (progressWrap) {
            progressWrap.classList.remove('d-none');
        }
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
        }

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/datasets/upload');
        xhr.upload.addEventListener('progress', (event) => {
            if (!event.lengthComputable || !progressBar) {
                return;
            }
            const percent = Math.round((event.loaded / event.total) * 100);
            progressBar.style.width = `${percent}%`;
            progressBar.textContent = `${percent}%`;
        });
        xhr.onload = () => {
            const payload = JSON.parse(xhr.responseText || '{}');
            if (xhr.status >= 200 && xhr.status < 300) {
                if (status) {
                    status.textContent = `Loaded ${payload.dataset_label}. Redirecting to dashboard...`;
                }
                window.location.href = payload.redirect_url;
                return;
            }
            if (status) {
                status.textContent = payload.error || 'Upload failed.';
            }
            showMessage(payload.error || 'Upload failed.');
        };
        xhr.onerror = () => {
            if (status) {
                status.textContent = 'Upload failed.';
            }
            showMessage('Upload failed.');
        };
        xhr.send(formData);
    };

    if (!input) {
        return;
    }
    input.addEventListener('change', async () => {
        const file = input.files?.[0];
        if (!file) {
            return;
        }
        uploadFile(file);
    });

    if (dropzone) {
        ['dragenter', 'dragover'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add('is-dragover');
            });
        });
        ['dragleave', 'drop'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove('is-dragover');
            });
        });
        dropzone.addEventListener('drop', (event) => {
            const file = event.dataTransfer?.files?.[0];
            if (file) {
                uploadFile(file);
            }
        });
    }
}

function currentDatasetKey() {
    return window.APP_CONTEXT?.datasetKey || document.body.dataset.datasetKey || 'iris';
}

function withDatasetQuery(url) {
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}dataset=${encodeURIComponent(currentDatasetKey())}`;
}

function showMessage(message, variant = 'error') {
    const element = document.getElementById('global-message');
    if (!element) {
        return;
    }
    element.textContent = message;
    element.className = `global-message ${variant}`;
    element.classList.remove('d-none');
    window.clearTimeout(showMessage.timeoutId);
    showMessage.timeoutId = window.setTimeout(() => {
        element.classList.add('d-none');
    }, 3600);
}

function renderPlot(elementId, figure) {
    const element = document.getElementById(elementId);
    if (!element || !figure) {
        return;
    }
    Plotly.react(element, figure.data, figure.layout, {
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
    });
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            'Content-Type': 'application/json',
        },
        ...options,
    });
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error || 'Request failed.');
    }
    return payload;
}

function createMetricCards(metrics, keyMap) {
    return keyMap.map(([key, label]) => `
        <div class="mini-metric-card">
            <span class="label">${label}</span>
            <span class="value">${metrics[key] ?? '-'}</span>
        </div>
    `).join('');
}

function renderReportTable(elementId, rows, extraColumns = []) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    const headers = ['Label', 'Precision', 'Recall', 'F1 score', 'Support', ...extraColumns.map((column) => column.label)];
    const body = rows.map((row) => `
        <tr>
            <td>${row.label || row.feature || ''}</td>
            <td>${row.precision ?? '-'}</td>
            <td>${row.recall ?? '-'}</td>
            <td>${row.f1_score ?? '-'}</td>
            <td>${row.support ?? '-'}</td>
            ${extraColumns.map((column) => `<td>${row[column.key] ?? '-'}</td>`).join('')}
        </tr>
    `).join('');
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table report-table align-middle mb-0">
                <thead><tr>${headers.map((header) => `<th>${header}</th>`).join('')}</tr></thead>
                <tbody>${body}</tbody>
            </table>
        </div>
    `;
}

function renderComparisonTable(rows) {
    const element = document.getElementById('comparison-table');
    if (!element) {
        return;
    }
    const body = rows.map((row) => `
        <tr>
            <td>${row.model}</td>
            <td>${row.accuracy}</td>
            <td>${row.precision}</td>
            <td>${row.recall}</td>
            <td>${row.f1_score}</td>
            <td>${row.cv_accuracy}</td>
            <td>${row.train_accuracy}</td>
        </tr>
    `).join('');
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table comparison-table align-middle mb-0">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>CV Accuracy</th>
                        <th>Train Accuracy</th>
                    </tr>
                </thead>
                <tbody>${body}</tbody>
            </table>
        </div>
    `;
}

function renderOriginalDatasetTable(rows) {
    const element = document.getElementById('bootstrap-original-table');
    if (!element || !rows.length) {
        return;
    }
    const headers = Object.keys(rows[0].features || {});
    const body = rows.map((row) => `
        <tr>
            <td>${row.row_id}</td>
            ${headers.map((header) => `<td>${row.features[header]}</td>`).join('')}
            <td>${row.species}</td>
        </tr>
    `).join('');
    element.innerHTML = `
        <table class="table dataset-table align-middle mb-0">
            <thead>
                <tr>
                    <th>Row</th>
                    ${headers.map((header) => `<th>${header}</th>`).join('')}
                    <th>Species</th>
                </tr>
            </thead>
            <tbody>${body}</tbody>
        </table>
    `;
}

function renderBootstrapSamples(sampleSummaries) {
    const element = document.getElementById('bootstrap-samples');
    if (!element) {
        return;
    }
    element.innerHTML = `<div class="bootstrap-sample-grid">${sampleSummaries.map((sample) => `
        <div class="bootstrap-sample-card">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <strong>Bootstrap Sample ${sample.sample_number}</strong>
                <span class="inline-badge">Repeated rows: ${sample.repeated_indices.length}</span>
            </div>
            <div class="small text-muted mb-2">Unique observations: ${sample.unique_count} | OOB observations: ${sample.oob_count}</div>
            <div class="small mb-2"><strong>Included:</strong> ${sample.included_pct}% | <strong>Excluded:</strong> ${sample.excluded_pct}%</div>
            <div class="small mb-2"><strong>Repeated indices:</strong> ${sample.repeated_indices.slice(0, 12).join(', ') || 'None'}</div>
            <div class="small"><strong>OOB indices:</strong> ${sample.oob_indices.slice(0, 12).join(', ') || 'None'}</div>
        </div>
    `).join('')}</div>`;
}

function stopBootstrapAnimation() {
    if (bootstrapAnimationId) {
        window.clearInterval(bootstrapAnimationId);
        bootstrapAnimationId = null;
    }
    const playButton = document.getElementById('bootstrap-play');
    if (playButton) {
        playButton.textContent = 'Play';
    }
}

function renderBootstrapFocus(sample) {
    if (!sample) {
        return;
    }
    const colors = sample.counts.map((count) => {
        if (count === 0) {
            return '#cbd5e1';
        }
        if (count > 1) {
            return '#f97316';
        }
        return '#2563eb';
    });
    renderPlot('bootstrap-focus-chart', {
        data: [{
            type: 'bar',
            x: sample.counts.map((_, index) => index),
            y: sample.counts,
            marker: { color: colors },
            hovertemplate: 'Observation %{x}<br>Count %{y}<extra></extra>',
        }],
        layout: {
            title: `Bootstrap Sample ${sample.sample_number} Frequency Profile`,
            margin: { l: 40, r: 20, t: 55, b: 40 },
            xaxis: { title: 'Observation index' },
            yaxis: { title: 'Count in current sample' },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
        },
    });
    const label = document.getElementById('bootstrap-step-label');
    if (label) {
        label.textContent = `Sample ${sample.sample_number} | Unique ${sample.unique_count} | OOB ${sample.oob_count}`;
    }
}

function bootstrapStepAt(position) {
    if (!currentBootstrapData || !currentBootstrapData.sample_summaries.length) {
        return;
    }
    const sample = currentBootstrapData.sample_summaries[position - 1];
    renderBootstrapFocus(sample);
    const slider = document.getElementById('bootstrap-step');
    if (slider) {
        slider.value = String(position);
    }
}

function renderOverview(data) {
    renderPlot('overview-comparison-chart', data.charts.comparison);
    if (data.profile_charts) {
        renderPlot('overview-target-distribution-chart', data.profile_charts.target_distribution);
        renderPlot('overview-missing-values-chart', data.profile_charts.missing_values);
        renderPlot('overview-correlation-chart', data.profile_charts.correlation_heatmap);
    }
    renderBackendRunHistory('overview-run-history', data.run_history || []);
}

function renderProfileDtypeSummary(rows) {
    const element = document.getElementById('profile-dtype-summary');
    if (!element) {
        return;
    }
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead>
                <tbody>
                    ${rows.map((row) => `<tr><td>${row.column}</td><td>${row.dtype}</td><td>${row.missing}</td></tr>`).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderColumnSummary(summary) {
    const element = document.getElementById('profile-column-summary');
    if (!element) {
        return;
    }
    const rows = Object.entries(summary)
        .filter(([key]) => key !== 'top_values')
        .map(([key, value]) => `<tr><td>${key}</td><td>${value ?? '-'}</td></tr>`)
        .join('');
    const topValues = summary.top_values?.length
        ? `<div class="mt-3"><strong>Top values</strong><div class="small mt-2">${summary.top_values.map((row) => `${row.label}: ${row.count}`).join('<br>')}</div></div>`
        : '';
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead><tr><th>Property</th><th>Value</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        ${topValues}
    `;
}

function renderDatasetProfilePage(data) {
    renderPlot('profile-target-distribution-chart', data.charts.target_distribution);
    renderPlot('profile-missing-values-chart', data.charts.missing_values);
    renderPlot('profile-correlation-chart', data.charts.correlation_heatmap);
    renderPlot('profile-column-chart', data.charts.column_distribution);
    renderColumnSummary(data.column_profile);
    renderProfileDtypeSummary(data.dtype_summary);
}

function renderExperimentsHistory(rows) {
    const element = document.getElementById('experiments-history-table');
    if (!element) {
        return;
    }
    if (!rows.length) {
        element.innerHTML = '<p class="text-muted mb-0">No persistent runs recorded for this dataset yet.</p>';
        return;
    }
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead><tr><th>Timestamp</th><th>Model</th><th>Metrics</th><th>Notes</th><th>Save</th></tr></thead>
                <tbody>
                    ${rows.map((row) => `
                        <tr data-run-id="${row.id}">
                            <td>${row.timestamp}</td>
                            <td>${row.model_type || '-'}</td>
                            <td><pre class="mb-0">${JSON.stringify(row.metrics || {}, null, 2)}</pre></td>
                            <td><textarea class="form-control experiment-note" rows="3">${row.notes || ''}</textarea></td>
                            <td><button class="btn btn-sm btn-outline-dark experiment-note-save">Save note</button></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderExperimentSelectors(rows) {
    const options = rows.map((row) => `<option value="${row.id}">${row.timestamp} • ${row.model_type}</option>`).join('');
    const selectA = document.getElementById('experiment-select-a');
    const selectB = document.getElementById('experiment-select-b');
    if (selectA) {
        selectA.innerHTML = options;
    }
    if (selectB) {
        selectB.innerHTML = options;
        if (rows.length > 1) {
            selectB.selectedIndex = 1;
        }
    }
}

function renderExperimentDiff(rows) {
    const selectA = document.getElementById('experiment-select-a');
    const selectB = document.getElementById('experiment-select-b');
    const element = document.getElementById('experiments-diff');
    if (!selectA || !selectB || !element) {
        return;
    }
    const left = rows.find((row) => row.id === selectA.value);
    const right = rows.find((row) => row.id === selectB.value);
    if (!left || !right) {
        element.innerHTML = '<p class="text-muted mb-0">Select two runs to compare.</p>';
        return;
    }
    element.innerHTML = `
        <div class="row g-4">
            <div class="col-lg-6"><div class="surface-subcard"><h4 class="mb-3">Run A</h4><pre class="mb-0">${JSON.stringify(left, null, 2)}</pre></div></div>
            <div class="col-lg-6"><div class="surface-subcard"><h4 class="mb-3">Run B</h4><pre class="mb-0">${JSON.stringify(right, null, 2)}</pre></div></div>
        </div>
    `;
}

function renderBackendRunHistory(elementId, rows) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    if (!rows.length) {
        element.innerHTML = '<p class="text-muted mb-0">No saved runs yet for this dataset.</p>';
        return;
    }
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Model</th>
                        <th>Hyperparameters</th>
                        <th>Metrics</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map((row) => `
                        <tr>
                            <td>${row.timestamp}</td>
                            <td>${row.model_type || '-'}</td>
                            <td><pre class="mb-0">${JSON.stringify(row.hyperparameters || {}, null, 2)}</pre></td>
                            <td><pre class="mb-0">${JSON.stringify(row.metrics || {}, null, 2)}</pre></td>
                            <td>${row.notes || '-'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

async function saveRunHistoryBackend(entry) {
    try {
        await requestJson('/api/run-history', {
            method: 'POST',
            body: JSON.stringify(entry),
        });
    } catch (error) {
        console.error('Run history save failed', error);
    }
}

function renderBootstrap(data) {
    currentBootstrapData = data;
    const summary = document.getElementById('bootstrap-summary');
    if (summary) {
        summary.innerHTML = createMetricCards({
            mean_unique: data.mean_unique,
            mean_oob: data.mean_oob,
            dataset_size: data.dataset_size,
        }, [
            ['mean_unique', 'Average unique rows'],
            ['mean_oob', 'Average OOB rows'],
            ['dataset_size', 'Dataset size'],
        ]);
    }
    renderPlot('bootstrap-frequency-chart', data.charts.frequency);
    renderPlot('bootstrap-heatmap-chart', data.charts.heatmap);
    renderPlot('bootstrap-composition-chart', data.charts.composition);
    renderBootstrapSamples(data.sample_summaries);
    renderOriginalDatasetTable(data.original_rows);
    const slider = document.getElementById('bootstrap-step');
    if (slider) {
        slider.max = String(data.sample_summaries.length);
        slider.value = '1';
    }
    stopBootstrapAnimation();
    bootstrapStepAt(1);
}

function renderPathDetails(pathRows) {
    const element = document.getElementById('decision-tree-path');
    if (!element) {
        return;
    }
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead>
                    <tr>
                        <th>Node</th>
                        <th>Rule</th>
                        <th>Sample value</th>
                        <th>Threshold</th>
                    </tr>
                </thead>
                <tbody>
                    ${pathRows.map((row) => `
                        <tr>
                            <td>${row.node_id}</td>
                            <td>${row.rule}</td>
                            <td>${row.sample_value ?? '-'}</td>
                            <td>${row.threshold ?? '-'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function populateFeatureSelects(featureNames) {
    const selects = [document.getElementById('decision-boundary-x'), document.getElementById('decision-boundary-y')];
    selects.forEach((select) => {
        if (!select || select.options.length > 0) {
            return;
        }
        select.innerHTML = featureNames.map((name) => `<option value="${name}">${name}</option>`).join('');
    });
    if (selects[0]) {
        selects[0].value = featureNames[2] || featureNames[0];
    }
    if (selects[1]) {
        selects[1].value = featureNames[3] || featureNames[1];
    }
}

function getHistory(key) {
    try {
        return JSON.parse(localStorage.getItem(key) || '[]');
    } catch {
        return [];
    }
}

function saveHistory(key, entry) {
    const history = getHistory(key);
    history.unshift({ ...entry, timestamp: new Date().toLocaleString() });
    localStorage.setItem(key, JSON.stringify(history.slice(0, 8)));
}

function renderHistory(elementId, entries, columns) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    if (!entries.length) {
        element.innerHTML = '<p class="text-muted mb-0">No runs recorded yet.</p>';
        return;
    }
    element.innerHTML = `
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        ${columns.map((column) => `<th>${column.label}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${entries.map((entry) => `
                        <tr>
                            <td>${entry.timestamp}</td>
                            ${columns.map((column) => `<td>${entry[column.key] ?? '-'}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderPredictionPlaygroundForm(featureNames, featureDefaults) {
    const element = document.getElementById('prediction-playground');
    if (!element) {
        return;
    }
    element.innerHTML = `
        <form id="prediction-playground-form" class="row g-3">
            <div class="col-md-4">
                <label class="form-label">Model</label>
                <select class="form-select" name="model_type">
                    <option value="decision_tree">Decision Tree</option>
                    <option value="bagging">Bagging</option>
                    <option value="random_forest" selected>Random Forest</option>
                </select>
            </div>
            ${featureNames.map((featureName) => `
                <div class="col-md-4">
                    <label class="form-label">${featureName}</label>
                    <input class="form-control" type="number" step="0.1" name="${featureName}" value="${featureDefaults[featureName]}">
                </div>
            `).join('')}
            <div class="col-12 d-grid d-md-flex gap-2 justify-content-md-start">
                <button class="btn btn-primary action-btn" type="submit">Predict</button>
            </div>
        </form>
        <div id="prediction-playground-output" class="mt-4"></div>
    `;
}

function renderPredictionPlaygroundOutput(data) {
    const element = document.getElementById('prediction-playground-output');
    if (!element) {
        return;
    }
    element.innerHTML = `
        <div class="row g-3 mb-3">
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">Model</span><span class="value text-capitalize">${data.model_type.replace('_', ' ')}</span></div></div>
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">Predicted class</span><span class="value">${data.predicted_label}</span></div></div>
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">Top signal</span><span class="value">${data.top_features[0]?.feature ?? '-'}</span></div></div>
        </div>
        <div id="prediction-playground-chart" class="plot-container"></div>
        <div class="table-responsive mt-3">
            <table class="table align-middle mb-0">
                <thead><tr><th>Feature</th><th>Global importance</th></tr></thead>
                <tbody>
                    ${data.top_features.map((row) => `<tr><td>${row.feature}</td><td>${row.importance}</td></tr>`).join('')}
                </tbody>
            </table>
        </div>
    `;
    renderPlot('prediction-playground-chart', data.charts.probabilities);
}

function getDecisionTreeFormPayload(form) {
    return {
        dataset_key: currentDatasetKey(),
        max_depth: form.max_depth.value === '' ? null : Number(form.max_depth.value),
        min_samples_leaf: Number(form.min_samples_leaf.value),
        random_state: Number(form.random_state.value),
    };
}

async function loadDecisionTreeExplorer() {
    const form = document.getElementById('decision-tree-form');
    if (!form) {
        return;
    }
    const payload = {
        ...getDecisionTreeFormPayload(form),
        sample_index: Number(document.getElementById('decision-tree-sample-index')?.value || 0),
        feature_x: document.getElementById('decision-boundary-x')?.value,
        feature_y: document.getElementById('decision-boundary-y')?.value,
    };
    const data = await requestJson('/api/decision-tree/explorer', { method: 'POST', body: JSON.stringify(payload) });
    renderPlot('decision-tree-probability-chart', data.charts.probabilities);
    renderPlot('decision-tree-boundary-chart', data.charts.decision_boundary);
    renderPathDetails(data.path_rows);
}

function renderDecisionTree(data) {
    currentDecisionTreeData = data;
    const metricsElement = document.getElementById('decision-tree-metrics');
    if (metricsElement) {
        metricsElement.innerHTML = createMetricCards(data.metrics, [
            ['train_accuracy', 'Train accuracy'],
            ['test_accuracy', 'Test accuracy'],
            ['cv_accuracy', 'CV accuracy'],
            ['precision', 'Precision'],
            ['recall', 'Recall'],
            ['f1_score', 'F1 score'],
        ]);
    }
    renderPlot('decision-tree-metric-chart', data.charts.metric_snapshot);
    renderPlot('decision-tree-confusion-chart', data.charts.confusion_matrix);
    renderPlot('decision-tree-importance-chart', data.charts.feature_importance);
    renderPlot('decision-tree-prediction-chart', data.charts.prediction_distribution);
    renderReportTable('decision-tree-report', data.metrics.classification_rows);
    const rulesElement = document.getElementById('decision-tree-rules');
    if (rulesElement) {
        rulesElement.textContent = data.tree_rules;
    }
    const sampleInput = document.getElementById('decision-tree-sample-index');
    if (sampleInput) {
        sampleInput.max = String(Math.max(data.sample_count - 1, 0));
    }
    populateFeatureSelects(data.feature_names);
    renderPredictionPlaygroundForm(data.feature_names, data.feature_defaults);
    bindPredictionPlayground();
    renderHistory('decision-tree-history', getHistory(HISTORY_KEYS.decisionTree), [
        { key: 'max_depth', label: 'max_depth' },
        { key: 'min_samples_leaf', label: 'min_samples_leaf' },
        { key: 'test_accuracy', label: 'Test accuracy' },
        { key: 'f1_score', label: 'F1' },
    ]);
}

function renderBagging(data) {
    const element = document.getElementById('bagging-metrics');
    if (element) {
        element.innerHTML = createMetricCards(data.metrics, [
            ['train_accuracy', 'Train accuracy'],
            ['test_accuracy', 'Test accuracy'],
            ['cv_accuracy', 'CV accuracy'],
            ['precision', 'Precision'],
            ['recall', 'Recall'],
            ['f1_score', 'F1 score'],
            ['oob_score', 'OOB score'],
            ['oob_error', 'OOB error'],
        ]);
    }
    renderPlot('bagging-accuracy-chart', data.charts.accuracy_curve);
    renderPlot('bagging-confusion-chart', data.charts.confusion_matrix);
    renderPlot('bagging-metric-chart', data.charts.metric_snapshot);
    renderPlot('bagging-prediction-chart', data.charts.prediction_distribution);
    renderReportTable('bagging-report', data.metrics.classification_rows);
}

function renderVotingDetails(data) {
    const element = document.getElementById('random-forest-voting-details');
    if (!element) {
        return;
    }
    const rows = Object.entries(data.sample_features).map(([feature, value]) => `<tr><td>${feature}</td><td>${value}</td></tr>`).join('');
    element.innerHTML = `
        <div class="row g-3 mb-3">
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">True label</span><span class="value">${data.true_label}</span></div></div>
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">Predicted label</span><span class="value">${data.predicted_label}</span></div></div>
            <div class="col-md-4"><div class="mini-metric-card"><span class="label">Sample index</span><span class="value">${data.sample_index}</span></div></div>
        </div>
        <div class="table-responsive">
            <table class="table align-middle mb-0">
                <thead><tr><th>Feature</th><th>Value</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

async function loadRandomForestVoting() {
    const form = document.getElementById('random-forest-form');
    if (!form) {
        return;
    }
    const payload = {
        n_estimators: Number(form.n_estimators.value),
        max_depth: form.max_depth.value === '' ? null : Number(form.max_depth.value),
        min_samples_split: Number(form.min_samples_split.value),
        min_samples_leaf: Number(form.min_samples_leaf.value),
        max_features: form.max_features.value,
        random_state: Number(form.random_state.value),
        sample_index: Number(document.getElementById('rf-sample-index')?.value || 0),
    };
    const data = await requestJson('/api/random-forest/voting', { method: 'POST', body: JSON.stringify(payload) });
    renderPlot('random-forest-vote-count-chart', data.charts.vote_counts);
    renderPlot('random-forest-tree-vote-chart', data.charts.tree_votes);
    renderPlot('random-forest-probability-chart', data.charts.probabilities);
    renderVotingDetails(data);
}

function renderRandomForest(data) {
    currentRandomForestData = data;
    const element = document.getElementById('random-forest-metrics');
    if (element) {
        element.innerHTML = createMetricCards(data.metrics, [
            ['train_accuracy', 'Train accuracy'],
            ['test_accuracy', 'Test accuracy'],
            ['cv_accuracy', 'CV accuracy'],
            ['precision', 'Precision'],
            ['recall', 'Recall'],
            ['f1_score', 'F1 score'],
            ['oob_score', 'OOB score'],
            ['oob_error', 'OOB error'],
        ]);
    }
    renderPlot('random-forest-accuracy-chart', data.charts.accuracy_curve);
    renderPlot('random-forest-confusion-chart', data.charts.confusion_matrix);
    renderPlot('random-forest-importance-chart', data.charts.feature_importance);
    renderPlot('random-forest-prediction-chart', data.charts.prediction_distribution);
    renderPlot('random-forest-metric-chart', data.charts.metric_snapshot);
    renderReportTable('random-forest-report', data.metrics.classification_rows);
    const sampleInput = document.getElementById('rf-sample-index');
    if (sampleInput) {
        sampleInput.max = String(Math.max(data.sample_count - 1, 0));
    }
    renderHistory('random-forest-history', getHistory(HISTORY_KEYS.randomForest), [
        { key: 'n_estimators', label: 'Estimators' },
        { key: 'max_depth', label: 'max_depth' },
        { key: 'test_accuracy', label: 'Test accuracy' },
        { key: 'oob_score', label: 'OOB score' },
    ]);
}

function renderComparison(data) {
    renderPlot('comparison-chart', data.charts.comparison);
    renderComparisonTable(data.rows);
    const combined = [...getHistory(HISTORY_KEYS.decisionTree), ...getHistory(HISTORY_KEYS.randomForest)]
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 10);
    renderHistory('comparison-history', combined, [
        { key: 'model', label: 'Model' },
        { key: 'test_accuracy', label: 'Test accuracy' },
        { key: 'f1_score', label: 'F1' },
        { key: 'oob_score', label: 'OOB score' },
    ]);
}

function renderImportance(data, activeView = 'gini') {
    const hasChart = data.charts && data.charts[activeView];
    if (hasChart) {
        renderPlot('importance-chart', data.charts[activeView]);
    }
    const tableElement = document.getElementById('importance-table');
    if (tableElement) {
        const rows = activeView === 'gini' ? data.gini_rows : data.permutation_rows;
        if (activeView === 'permutation' && !data.has_permutation) {
            tableElement.innerHTML = '<p class="text-muted mb-0">Permutation importance has not been computed yet. Use the controls or the permutation tab to load it.</p>';
        } else {
        const extraHeaders = activeView === 'permutation'
            ? [{ key: 'importance', label: 'Mean importance' }, { key: 'std', label: 'Std dev' }]
            : [{ key: 'importance', label: 'Importance' }];
        tableElement.innerHTML = `
            <div class="table-responsive">
                <table class="table align-middle mb-0">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            ${extraHeaders.map((header) => `<th>${header.label}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map((row) => `
                            <tr>
                                <td>${row.feature}</td>
                                ${extraHeaders.map((header) => `<td>${row[header.key]}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        }
    }
    const giniButton = document.getElementById('importance-toggle-gini');
    const permutationButton = document.getElementById('importance-toggle-permutation');
    if (giniButton && permutationButton) {
        giniButton.className = activeView === 'gini' ? 'btn btn-dark active' : 'btn btn-outline-dark';
        permutationButton.className = activeView === 'permutation' ? 'btn btn-dark active' : 'btn btn-outline-dark';
    }
}

function renderOob(data) {
    const element = document.getElementById('oob-metrics');
    if (element) {
        element.innerHTML = createMetricCards(data, [
            ['oob_score', 'OOB score'],
            ['oob_error', 'OOB error'],
            ['n_estimators', 'Estimators'],
            ['max_features', 'max_features'],
        ]);
    }
    renderPlot('oob-chart', data.charts.oob_curve);
}

function renderBiasVariance(data) {
    renderPlot('bias-variance-chart', data.charts.bias_variance);
    renderPlot('generalization-gap-chart', data.charts.generalization_gap);
}

function bindPredictionPlayground() {
    const form = document.getElementById('prediction-playground-form');
    if (!form || form.dataset.bound === 'true') {
        return;
    }
    form.dataset.bound = 'true';
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const features = {};
            Object.keys(currentDecisionTreeData.feature_defaults).forEach((featureName) => {
                features[featureName] = Number(form[featureName].value);
            });
            const data = await requestJson('/api/predict', {
                method: 'POST',
                body: JSON.stringify({
                    model_type: form.model_type.value,
                    dataset_key: currentDatasetKey(),
                    random_state: Number(document.getElementById('decision-tree-form').random_state.value),
                    features,
                }),
            });
            renderPredictionPlaygroundOutput(data);
            showMessage('Prediction playground updated.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindBootstrapPage() {
    renderBootstrap(window.INITIAL_PAGE_DATA);
    const form = document.getElementById('bootstrap-form');
    const slider = document.getElementById('bootstrap-step');
    const playButton = document.getElementById('bootstrap-play');
    if (slider) {
        slider.addEventListener('input', () => {
            stopBootstrapAnimation();
            bootstrapStepAt(Number(slider.value));
        });
    }
    if (playButton) {
        playButton.addEventListener('click', () => {
            if (!currentBootstrapData) {
                return;
            }
            if (bootstrapAnimationId) {
                stopBootstrapAnimation();
                return;
            }
            playButton.textContent = 'Pause';
            bootstrapAnimationId = window.setInterval(() => {
                const current = Number(document.getElementById('bootstrap-step')?.value || 1);
                const next = current >= currentBootstrapData.sample_summaries.length ? 1 : current + 1;
                bootstrapStepAt(next);
            }, 1200);
        });
    }
    if (!form) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const payload = {
                dataset_key: currentDatasetKey(),
                sample_size: Number(form.sample_size.value),
                n_samples: Number(form.n_samples.value),
                random_seed: Number(form.random_seed.value),
            };
            const data = await requestJson('/api/bootstrap', { method: 'POST', body: JSON.stringify(payload) });
            renderBootstrap(data);
            showMessage('Bootstrap samples updated.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindDecisionTreePage() {
    renderDecisionTree(window.INITIAL_PAGE_DATA);
    loadDecisionTreeExplorer().catch((error) => showMessage(error.message));
    const form = document.getElementById('decision-tree-form');
    const tuneButton = document.getElementById('decision-tree-tune');
    const exploreButton = document.getElementById('decision-tree-explore');
    if (exploreButton) {
        exploreButton.addEventListener('click', async () => {
            try {
                await loadDecisionTreeExplorer();
                showMessage('Decision Tree explorer updated.', 'success');
            } catch (error) {
                showMessage(error.message);
            }
        });
    }
    if (!form || !tuneButton) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const payload = getDecisionTreeFormPayload(form);
            const data = await requestJson('/api/decision-tree/train', { method: 'POST', body: JSON.stringify(payload) });
            renderDecisionTree(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Decision Tree',
                hyperparameters: {
                    max_depth: payload.max_depth,
                    min_samples_leaf: payload.min_samples_leaf,
                    random_state: payload.random_state,
                },
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            saveHistory(HISTORY_KEYS.decisionTree, {
                model: 'Decision Tree',
                max_depth: payload.max_depth ?? 'None',
                min_samples_leaf: payload.min_samples_leaf,
                test_accuracy: data.metrics.test_accuracy,
                f1_score: data.metrics.f1_score,
            });
            renderHistory('decision-tree-history', getHistory(HISTORY_KEYS.decisionTree), [
                { key: 'max_depth', label: 'max_depth' },
                { key: 'min_samples_leaf', label: 'min_samples_leaf' },
                { key: 'test_accuracy', label: 'Test accuracy' },
                { key: 'f1_score', label: 'F1' },
            ]);
            await loadDecisionTreeExplorer();
            showMessage('Decision Tree retrained.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
    tuneButton.addEventListener('click', async () => {
        try {
            const randomState = Number(form.random_state.value);
            const data = await requestJson('/api/decision-tree/tune', { method: 'POST', body: JSON.stringify({ dataset_key: currentDatasetKey(), random_state: randomState }) });
            renderDecisionTree(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Decision Tree',
                hyperparameters: data.best_params,
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            form.max_depth.value = data.best_params.max_depth ?? '';
            form.min_samples_leaf.value = data.best_params.min_samples_leaf;
            saveHistory(HISTORY_KEYS.decisionTree, {
                model: 'Decision Tree',
                max_depth: data.best_params.max_depth ?? 'None',
                min_samples_leaf: data.best_params.min_samples_leaf,
                test_accuracy: data.metrics.test_accuracy,
                f1_score: data.metrics.f1_score,
            });
            renderHistory('decision-tree-history', getHistory(HISTORY_KEYS.decisionTree), [
                { key: 'max_depth', label: 'max_depth' },
                { key: 'min_samples_leaf', label: 'min_samples_leaf' },
                { key: 'test_accuracy', label: 'Test accuracy' },
                { key: 'f1_score', label: 'F1' },
            ]);
            await loadDecisionTreeExplorer();
            showMessage('Decision Tree hyperparameters tuned from notebook grid.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindBaggingPage() {
    renderBagging(window.INITIAL_PAGE_DATA);
    const form = document.getElementById('bagging-form');
    const tuneButton = document.getElementById('bagging-tune');
    if (!form || !tuneButton) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const payload = {
                dataset_key: currentDatasetKey(),
                n_estimators: Number(form.n_estimators.value),
                max_samples: Number(form.max_samples.value),
                max_features: Number(form.max_features.value),
                random_state: Number(form.random_state.value),
                max_depth: form.max_depth.value === '' ? null : Number(form.max_depth.value),
                min_samples_leaf: Number(form.min_samples_leaf.value),
            };
            const data = await requestJson('/api/bagging/train', { method: 'POST', body: JSON.stringify(payload) });
            renderBagging(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Bagging',
                hyperparameters: {
                    n_estimators: payload.n_estimators,
                    max_samples: payload.max_samples,
                    max_features: payload.max_features,
                    max_depth: payload.max_depth,
                    min_samples_leaf: payload.min_samples_leaf,
                    random_state: payload.random_state,
                },
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            showMessage('Bagging model retrained.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
    tuneButton.addEventListener('click', async () => {
        try {
            const data = await requestJson('/api/bagging/tune', { method: 'POST', body: JSON.stringify({ dataset_key: currentDatasetKey(), random_state: Number(form.random_state.value) }) });
            renderBagging(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Bagging',
                hyperparameters: data.best_params,
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            form.n_estimators.value = data.best_params.n_estimators;
            form.max_samples.value = data.best_params.max_samples;
            form.max_features.value = data.best_params.max_features;
            showMessage('Bagging hyperparameters tuned from notebook grid.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindRandomForestPage() {
    renderRandomForest(window.INITIAL_PAGE_DATA);
    loadRandomForestVoting().catch((error) => showMessage(error.message));
    const form = document.getElementById('random-forest-form');
    const tuneButton = document.getElementById('random-forest-tune');
    const votingButton = document.getElementById('random-forest-voting');
    if (votingButton) {
        votingButton.addEventListener('click', async () => {
            try {
                await loadRandomForestVoting();
                showMessage('Random Forest voting explorer updated.', 'success');
            } catch (error) {
                showMessage(error.message);
            }
        });
    }
    if (!form || !tuneButton) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const payload = {
                dataset_key: currentDatasetKey(),
                dataset_key: currentDatasetKey(),
                n_estimators: Number(form.n_estimators.value),
                max_depth: form.max_depth.value === '' ? null : Number(form.max_depth.value),
                min_samples_split: Number(form.min_samples_split.value),
                min_samples_leaf: Number(form.min_samples_leaf.value),
                max_features: form.max_features.value,
                random_state: Number(form.random_state.value),
                oob_enabled: form.oob_enabled.checked,
            };
            const data = await requestJson('/api/random-forest/train', { method: 'POST', body: JSON.stringify(payload) });
            renderRandomForest(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Random Forest',
                hyperparameters: {
                    n_estimators: payload.n_estimators,
                    max_depth: payload.max_depth,
                    min_samples_split: payload.min_samples_split,
                    min_samples_leaf: payload.min_samples_leaf,
                    max_features: payload.max_features,
                    bootstrap: true,
                    oob_enabled: payload.oob_enabled,
                    random_state: payload.random_state,
                },
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            saveHistory(HISTORY_KEYS.randomForest, {
                model: 'Random Forest',
                n_estimators: payload.n_estimators,
                max_depth: payload.max_depth ?? 'None',
                test_accuracy: data.metrics.test_accuracy,
                f1_score: data.metrics.f1_score,
                oob_score: data.metrics.oob_score ?? '-',
            });
            renderHistory('random-forest-history', getHistory(HISTORY_KEYS.randomForest), [
                { key: 'n_estimators', label: 'Estimators' },
                { key: 'max_depth', label: 'max_depth' },
                { key: 'test_accuracy', label: 'Test accuracy' },
                { key: 'oob_score', label: 'OOB score' },
            ]);
            await loadRandomForestVoting();
            showMessage('Random Forest retrained.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
    tuneButton.addEventListener('click', async () => {
        try {
            const data = await requestJson('/api/random-forest/tune', { method: 'POST', body: JSON.stringify({ dataset_key: currentDatasetKey(), random_state: Number(form.random_state.value) }) });
            renderRandomForest(data);
            await saveRunHistoryBackend({
                dataset_key: currentDatasetKey(),
                dataset_label: document.getElementById('dataset-switcher')?.selectedOptions?.[0]?.textContent || currentDatasetKey(),
                model_type: 'Random Forest',
                hyperparameters: data.best_params,
                metrics: data.metrics,
                notes: form.notes?.value || '',
            });
            form.n_estimators.value = data.best_params.n_estimators;
            form.max_depth.value = data.best_params.max_depth ?? '';
            form.min_samples_leaf.value = data.best_params.min_samples_leaf;
            form.max_features.value = data.best_params.max_features;
            form.oob_enabled.checked = true;
            saveHistory(HISTORY_KEYS.randomForest, {
                model: 'Random Forest',
                n_estimators: data.best_params.n_estimators,
                max_depth: data.best_params.max_depth ?? 'None',
                test_accuracy: data.metrics.test_accuracy,
                f1_score: data.metrics.f1_score,
                oob_score: data.metrics.oob_score ?? '-',
            });
            renderHistory('random-forest-history', getHistory(HISTORY_KEYS.randomForest), [
                { key: 'n_estimators', label: 'Estimators' },
                { key: 'max_depth', label: 'max_depth' },
                { key: 'test_accuracy', label: 'Test accuracy' },
                { key: 'oob_score', label: 'OOB score' },
            ]);
            await loadRandomForestVoting();
            showMessage('Random Forest hyperparameters tuned from notebook grid.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindComparisonPage() {
    renderComparison(window.INITIAL_PAGE_DATA);
    const form = document.getElementById('comparison-form');
    if (!form) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const query = new URLSearchParams({ random_state: String(Number(form.random_state.value)) });
            const data = await requestJson(withDatasetQuery(`/api/model-comparison?${query.toString()}`));
            renderComparison(data);
            showMessage('Comparison refreshed.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindImportancePage() {
    let activeView = 'gini';
    let currentData = window.INITIAL_PAGE_DATA;
    renderImportance(currentData, activeView);
    const form = document.getElementById('importance-form');
    const giniButton = document.getElementById('importance-toggle-gini');
    const permutationButton = document.getElementById('importance-toggle-permutation');
    if (giniButton) {
        giniButton.addEventListener('click', () => {
            activeView = 'gini';
            renderImportance(currentData, activeView);
        });
    }
    if (permutationButton) {
        permutationButton.addEventListener('click', async () => {
            activeView = 'permutation';
            if (!currentData.has_permutation) {
                try {
                    const query = new URLSearchParams({
                        random_state: String(Number(form?.random_state.value || 42)),
                        n_repeats: String(Number(form?.n_repeats.value || 10)),
                        include_permutation: 'true',
                    });
                    currentData = await requestJson(withDatasetQuery(`/api/feature-importance?${query.toString()}`));
                    showMessage('Permutation importance loaded.', 'success');
                } catch (error) {
                    showMessage(error.message);
                    return;
                }
            }
            renderImportance(currentData, activeView);
        });
    }
    if (!form) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const query = new URLSearchParams({
                random_state: String(Number(form.random_state.value)),
                n_repeats: String(Number(form.n_repeats.value)),
                include_permutation: 'true',
            });
            currentData = await requestJson(withDatasetQuery(`/api/feature-importance?${query.toString()}`));
            renderImportance(currentData, activeView);
            showMessage('Feature importance recomputed.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindOobPage() {
    renderOob(window.INITIAL_PAGE_DATA);
    const form = document.getElementById('oob-form');
    if (!form) {
        return;
    }
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        try {
            const query = new URLSearchParams({
                random_state: String(Number(form.random_state.value)),
                max_estimators: String(Number(form.max_estimators.value)),
            });
            const data = await requestJson(withDatasetQuery(`/api/oob?${query.toString()}`));
            renderOob(data);
            showMessage('OOB analysis updated.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindBiasVariancePage() {
    renderBiasVariance(window.INITIAL_PAGE_DATA);
}

function bindDatasetProfilePage() {
    let currentData = window.INITIAL_PAGE_DATA;
    renderDatasetProfilePage(currentData);

    const profileSelect = document.getElementById('profile-column-select');
    const refreshButton = document.getElementById('profile-column-refresh');
    const targetForm = document.getElementById('target-column-form');

    const reloadProfile = async () => {
        const column = profileSelect?.value || '';
        const query = new URLSearchParams({ dataset: currentDatasetKey(), column });
        currentData = await requestJson(`/api/dataset-profile?${query.toString()}`);
        renderDatasetProfilePage(currentData);
    };

    if (refreshButton) {
        refreshButton.addEventListener('click', async () => {
            try {
                await reloadProfile();
                showMessage('Column profile updated.', 'success');
            } catch (error) {
                showMessage(error.message);
            }
        });
    }

    if (targetForm) {
        targetForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            try {
                const payload = {
                    dataset_key: currentDatasetKey(),
                    target_column: document.getElementById('target-column-select')?.value,
                };
                const data = await requestJson('/api/datasets/target-column', {
                    method: 'POST',
                    body: JSON.stringify(payload),
                });
                window.location.href = data.redirect_url;
            } catch (error) {
                showMessage(error.message);
            }
        });
    }
}

function bindExperimentsPage() {
    let rows = window.INITIAL_PAGE_DATA.rows || [];
    renderExperimentsHistory(rows);
    renderExperimentSelectors(rows);
    renderExperimentDiff(rows);

    const diffButton = document.getElementById('experiment-compare-btn');
    if (diffButton) {
        diffButton.addEventListener('click', () => renderExperimentDiff(rows));
    }

    document.addEventListener('click', async (event) => {
        const button = event.target;
        if (!(button instanceof HTMLElement) || !button.classList.contains('experiment-note-save')) {
            return;
        }
        const rowElement = button.closest('[data-run-id]');
        const textarea = rowElement?.querySelector('.experiment-note');
        if (!rowElement || !(textarea instanceof HTMLTextAreaElement)) {
            return;
        }
        try {
            await requestJson(`/api/run-history/${rowElement.getAttribute('data-run-id')}`, {
                method: 'PUT',
                body: JSON.stringify({ notes: textarea.value }),
            });
            const response = await requestJson(`/api/run-history?dataset=${encodeURIComponent(currentDatasetKey())}&limit=50`);
            rows = response.rows || [];
            renderExperimentsHistory(rows);
            renderExperimentSelectors(rows);
            renderExperimentDiff(rows);
            showMessage('Run notes saved.', 'success');
        } catch (error) {
            showMessage(error.message);
        }
    });
}

function bindApiReferencePage() {
    const responseElement = document.getElementById('api-live-response');
    document.querySelectorAll('.api-copy-btn').forEach((button) => {
        button.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(button.dataset.copyText || '');
                showMessage('Endpoint details copied.', 'success');
            } catch {
                showMessage('Copy failed in this browser context.');
            }
        });
    });

    document.querySelectorAll('.api-try-btn').forEach((button) => {
        button.addEventListener('click', async () => {
            try {
                if (responseElement) {
                    responseElement.textContent = 'Loading live response...';
                }
                const endpoint = button.dataset.endpoint;
                const data = await requestJson(endpoint);
                if (responseElement) {
                    responseElement.textContent = JSON.stringify(data, null, 2);
                }
                showMessage('Live API response loaded.', 'success');
            } catch (error) {
                if (responseElement) {
                    responseElement.textContent = String(error.message);
                }
                showMessage(error.message);
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const datasetSwitcher = document.getElementById('dataset-switcher');
    if (datasetSwitcher) {
        datasetSwitcher.addEventListener('change', () => {
            const nextUrl = new URL(window.location.href);
            nextUrl.searchParams.set('dataset', datasetSwitcher.value);
            window.location.href = nextUrl.toString();
        });
    }
    const page = document.body.dataset.page;
    const binders = {
        home: bindHomePage,
        overview: () => renderOverview(window.INITIAL_PAGE_DATA),
        dataset_profile: bindDatasetProfilePage,
        bootstrap: bindBootstrapPage,
        decision_tree: bindDecisionTreePage,
        bagging: bindBaggingPage,
        random_forest: bindRandomForestPage,
        comparison: bindComparisonPage,
        importance: bindImportancePage,
        oob: bindOobPage,
        bias_variance: bindBiasVariancePage,
        experiments: bindExperimentsPage,
        api_reference: bindApiReferencePage,
    };
    const binder = binders[page];
    if (binder) {
        binder();
    }
});
