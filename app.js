/**
 * XAI U-Net Demo — Frontend Application
 *
 * Loads pre-computed demo data from data/samples.json,
 * renders the sample gallery and analysis viewer,
 * and handles all user interactions.
 */

(function () {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════════
    // Constants
    // ═══════════════════════════════════════════════════════════════════════

    const DATA_URL  = 'data/samples.json';
    const DATA_BASE = 'data/';  // image paths in JSON are relative to this

    // ═══════════════════════════════════════════════════════════════════════
    // State
    // ═══════════════════════════════════════════════════════════════════════

    let appData = null;          // Full manifest
    let samples = [];            // Array of sample objects
    let activeSampleId = null;   // Currently selected sample id
    let activeFilter = 'all';    // Current class filter

    // ═══════════════════════════════════════════════════════════════════════
    // DOM References
    // ═══════════════════════════════════════════════════════════════════════

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        navbar: $('#navbar'),
        sampleList: $('#sampleList'),
        filterTabs: $('#filterTabs'),
        emptyState: $('#emptyState'),
        analysisContent: $('#analysisContent'),
        // Header
        classBadge: $('#classBadge'),
        patientInfo: $('#patientInfo'),
        malignancyInfo: $('#malignancyInfo'),
        // Images
        imgCT: $('#imgCT'),
        imgPredOverlay: $('#imgPredOverlay'),
        imgGT: $('#imgGT'),
        imgIG: $('#imgIG'),
        imgIGOverlay: $('#imgIGOverlay'),
        imgIGOverlayZoomed: $('#imgIGOverlayZoomed'),
        // Panels
        panelIG: $('#panelIG'),
        panelIGOverlay: $('#panelIGOverlay'),
        panelIGOverlayZoomed: $('#panelIGOverlayZoomed'),
        // Metrics
        metricDice: $('#metricDice'),
        metricIoU: $('#metricIoU'),
        metricPrecision: $('#metricPrecision'),
        metricRecall: $('#metricRecall'),
        barDice: $('#barDice'),
        barIoU: $('#barIoU'),
        barPrecision: $('#barPrecision'),
        barRecall: $('#barRecall'),
        metricGTArea: $('#metricGTArea'),
        metricPredArea: $('#metricPredArea'),
        metricMaxConf: $('#metricMaxConf'),
        // Hero
        statSamples: $('#statSamples'),
        // Lightbox
        lightbox: $('#lightbox'),
        lightboxImg: $('#lightboxImg'),
        lightboxCaption: $('#lightboxCaption'),
        lightboxClose: $('#lightboxClose'),
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Data Loading
    // ═══════════════════════════════════════════════════════════════════════

    async function loadData() {
        try {
            const resp = await fetch(DATA_URL);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            appData = await resp.json();
            samples = appData.samples || [];
            console.log(`[XAI Demo] Loaded ${samples.length} samples`);
            return true;
        } catch (err) {
            console.warn('[XAI Demo] Could not load data:', err.message);
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Gallery
    // ═══════════════════════════════════════════════════════════════════════

    function renderSampleList() {
        const filtered = activeFilter === 'all'
            ? samples
            : samples.filter(s => s.class_name === activeFilter);

        if (filtered.length === 0) {
            els.sampleList.innerHTML = `
                <div class="error-state">
                    <h3>No Samples Found</h3>
                    <p>No samples match the current filter.</p>
                </div>
            `;
            return;
        }

        els.sampleList.innerHTML = filtered.map(sample => `
            <div class="sample-card ${sample.id === activeSampleId ? 'active' : ''}"
                 data-id="${sample.id}"
                 role="button"
                 tabindex="0"
                 aria-label="Sample ${sample.id}: Patient ${sample.patient_id}, ${sample.class_name}">
                <img class="sample-thumb"
                     src="${DATA_BASE}${sample.images.ct}"
                     alt="CT thumbnail"
                     loading="lazy"
                     onerror="this.style.display='none'">
                <div class="sample-meta">
                    <div class="sample-meta-top">
                        <span class="sample-id">P-${sample.patient_id}</span>
                        <span class="class-badge badge-${sample.class_name}">${sample.class_name}</span>
                    </div>
                    <span class="sample-dice">Dice: ${sample.metrics.dice.toFixed(3)}</span>
                </div>
            </div>
        `).join('');

        // Attach click handlers
        els.sampleList.querySelectorAll('.sample-card').forEach(card => {
            card.addEventListener('click', () => {
                const id = parseInt(card.dataset.id, 10);
                selectSample(id);
            });
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const id = parseInt(card.dataset.id, 10);
                    selectSample(id);
                }
            });
        });
    }

    function showErrorState() {
        els.sampleList.innerHTML = `
            <div class="error-state">
                <h3>No Data Found</h3>
                <p>Run the export script to generate demo data, then place the <code>data/</code> folder here.</p>
                <code>python export_demo_data.py --data-dir /path/to/output --output-dir ./data</code>
            </div>
        `;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Selection & Analysis Display
    // ═══════════════════════════════════════════════════════════════════════

    function selectSample(id) {
        activeSampleId = id;
        const sample = samples.find(s => s.id === id);
        if (!sample) return;

        // Update sidebar active state
        els.sampleList.querySelectorAll('.sample-card').forEach(card => {
            card.classList.toggle('active', parseInt(card.dataset.id, 10) === id);
        });

        // Show analysis content
        els.emptyState.style.display = 'none';
        els.analysisContent.style.display = 'block';

        // Re-trigger animation
        els.analysisContent.style.animation = 'none';
        /* eslint-disable-next-line no-unused-expressions */
        els.analysisContent.offsetHeight; // Force reflow
        els.analysisContent.style.animation = '';

        updateHeader(sample);
        updateImages(sample);
        updateMetrics(sample);
    }

    function updateHeader(sample) {
        // Class badge
        els.classBadge.textContent = sample.class_name;
        els.classBadge.className = `sample-badge badge-${sample.class_name}`;

        // Patient info
        els.patientInfo.textContent = `Patient #${sample.patient_id}`;

        // Malignancy
        const malScore = sample.malignancy_score || 0;
        let malLabel = 'N/A';
        if (malScore > 0) {
            malLabel = `${malScore}/5`;
            if (malScore <= 3) malLabel += ' (Low)';
            else malLabel += ' (High)';
        }
        els.malignancyInfo.textContent = `Malignancy: ${malLabel}`;

        // Add contextual explanation
        const explBox = $('#explanationBox');
        if (sample.class_name === 'Normal') {
            explBox.innerHTML = `<strong>Analysis:</strong> Since this is a Normal patient with no detected nodule, the ground truth and prediction masks are empty. Consequently, the Integrated Gradients (IG) attribution map is also mostly blank because the model does not identify any active features for a nodule prediction.`;
        } else if (sample.class_name === 'Benign' || sample.class_name === 'Malignant') {
            explBox.innerHTML = `<strong>Analysis:</strong> For this ${sample.class_name} case, the IG looks into the reason on <em>why</em> the model makes the prediction. The IG attribution map highlights the pixels that contributed most to the model's positive nodule prediction. If the IG heatmap aligns precisely with the nodule, it indicates the model is correctly predicting for the right reasons. If the focus is elsewhere, it may suggest shortcut learning or anomalies.`;
        } else {
            explBox.innerHTML = '';
        }
    }

    function updateImages(sample) {
        const images = sample.images;

        // Set image sources with error handling
        setImage(els.imgCT, images.ct, 'Original CT Scan');
        setImage(els.imgPredOverlay, images.pred_overlay, 'Prediction Overlay');
        setImage(els.imgGT, images.gt_overlay, 'Ground Truth Overlay');

        // IG images (may not exist)
        if (sample.has_ig && images.ig_heatmap) {
            setImage(els.imgIG, images.ig_heatmap, 'IG Attribution Map');
            els.panelIG.querySelector('.no-data-label')?.remove();
        } else {
            els.imgIG.src = '';
            els.imgIG.alt = '';
            if (!els.panelIG.querySelector('.no-data-label')) {
                const label = document.createElement('div');
                label.className = 'no-data-label';
                label.textContent = 'IG not computed';
                els.panelIG.querySelector('.viz-image-wrap').appendChild(label);
            }
        }

        if (sample.has_ig && images.ig_overlay) {
            setImage(els.imgIGOverlay, images.ig_overlay, 'IG + CT Overlay');
            els.panelIGOverlay.querySelector('.no-data-label')?.remove();
        } else {
            els.imgIGOverlay.src = '';
            els.imgIGOverlay.alt = '';
            if (!els.panelIGOverlay.querySelector('.no-data-label')) {
                const label = document.createElement('div');
                label.className = 'no-data-label';
                label.textContent = 'IG not computed';
                els.panelIGOverlay.querySelector('.viz-image-wrap').appendChild(label);
            }
        }

        if (sample.has_ig && images.ig_overlay_zoomed) {
            setImage(els.imgIGOverlayZoomed, images.ig_overlay_zoomed, 'IG + CT Overlay Zoomed');
            els.panelIGOverlayZoomed.querySelector('.no-data-label')?.remove();
        } else {
            els.imgIGOverlayZoomed.src = '';
            els.imgIGOverlayZoomed.alt = '';
            if (!els.panelIGOverlayZoomed.querySelector('.no-data-label')) {
                const label = document.createElement('div');
                label.className = 'no-data-label';
                label.textContent = 'No Zoom Available';
                els.panelIGOverlayZoomed.querySelector('.viz-image-wrap').appendChild(label);
            }
        }
    }

    function setImage(imgEl, src, alt) {
        if (!src) return;
        imgEl.src = DATA_BASE + src;
        imgEl.alt = alt;
        imgEl.onerror = function () {
            this.style.opacity = '0.3';
        };
    }

    function updateMetrics(sample) {
        const m = sample.metrics;

        // Animate metric values and bars
        animateMetric(els.metricDice, els.barDice, m.dice);
        animateMetric(els.metricIoU, els.barIoU, m.iou);
        animateMetric(els.metricPrecision, els.barPrecision, m.precision);
        animateMetric(els.metricRecall, els.barRecall, m.recall);

        // Extra metrics
        els.metricGTArea.textContent = m.gt_area?.toLocaleString() || '—';
        els.metricPredArea.textContent = m.pred_area?.toLocaleString() || '—';
        els.metricMaxConf.textContent = m.max_confidence?.toFixed(3) || '—';
    }

    function animateMetric(valueEl, barEl, value) {
        // Reset bar width first for re-animation
        barEl.style.width = '0%';

        valueEl.textContent = value.toFixed(4);

        // Trigger bar animation after a brief delay
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                barEl.style.width = `${Math.min(value * 100, 100)}%`;
            });
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Filter Tabs
    // ═══════════════════════════════════════════════════════════════════════

    function initFilters() {
        els.filterTabs.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                activeFilter = btn.dataset.filter;
                els.filterTabs.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderSampleList();
            });
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Lightbox (Image Zoom)
    // ═══════════════════════════════════════════════════════════════════════

    function initLightbox() {
        // Click on any viz image to open lightbox
        document.addEventListener('click', (e) => {
            const imgWrap = e.target.closest('.viz-image-wrap');
            if (!imgWrap) return;
            const img = imgWrap.querySelector('img');
            if (!img || !img.src) return;

            const label = imgWrap.closest('.viz-panel')?.querySelector('.viz-label');
            els.lightboxImg.src = img.src;
            els.lightboxCaption.textContent = label ? label.textContent : '';
            els.lightbox.classList.add('visible');
            document.body.style.overflow = 'hidden';
        });

        // Close lightbox
        const closeLightbox = () => {
            els.lightbox.classList.remove('visible');
            document.body.style.overflow = '';
        };

        els.lightboxClose.addEventListener('click', closeLightbox);
        els.lightbox.addEventListener('click', (e) => {
            if (e.target === els.lightbox) closeLightbox();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLightbox();
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Navbar Scroll Effect
    // ═══════════════════════════════════════════════════════════════════════

    function initNavScroll() {
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    els.navbar.classList.toggle('scrolled', window.scrollY > 50);
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Hero Stats Animation
    // ═══════════════════════════════════════════════════════════════════════

    function animateHeroStats() {
        $$('.stat-value[data-target]').forEach(el => {
            const target = parseInt(el.dataset.target, 10);
            animateCount(el, 0, target, 1200);
        });

        if (appData && appData.total_samples) {
            animateCount(els.statSamples, 0, appData.total_samples, 1000);
        }
    }

    function animateCount(el, start, end, duration) {
        const startTime = performance.now();

        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * ease);
            el.textContent = current.toLocaleString();
            if (progress < 1) requestAnimationFrame(update);
        }

        requestAnimationFrame(update);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Keyboard Navigation
    // ═══════════════════════════════════════════════════════════════════════

    function initKeyboardNav() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                e.preventDefault();
                navigateSample(1);
            } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                e.preventDefault();
                navigateSample(-1);
            }
        });
    }

    function navigateSample(direction) {
        if (samples.length === 0) return;

        const filtered = activeFilter === 'all'
            ? samples
            : samples.filter(s => s.class_name === activeFilter);

        if (filtered.length === 0) return;

        if (activeSampleId === null) {
            selectSample(filtered[0].id);
            return;
        }

        const currentIdx = filtered.findIndex(s => s.id === activeSampleId);
        let nextIdx = currentIdx + direction;
        if (nextIdx < 0) nextIdx = filtered.length - 1;
        if (nextIdx >= filtered.length) nextIdx = 0;

        selectSample(filtered[nextIdx].id);

        // Scroll the sidebar to show the active card
        const activeCard = els.sampleList.querySelector('.sample-card.active');
        if (activeCard) {
            activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Scroll Reveal Animations
    // ═══════════════════════════════════════════════════════════════════════

    function initScrollReveal() {
        const revealEls = $$('.method-card, .detail-card, .method-highlight, .arch-unet');

        if (!('IntersectionObserver' in window)) {
            revealEls.forEach(el => el.style.opacity = '1');
            return;
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

        revealEls.forEach((el, i) => {
            el.style.opacity = '0';
            el.style.animationDelay = `${i * 0.08}s`;
            observer.observe(el);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Initialization
    // ═══════════════════════════════════════════════════════════════════════

    async function init() {
        console.log('[XAI Demo] Initializing...');

        // Init UI features
        initNavScroll();
        initFilters();
        initLightbox();
        initKeyboardNav();
        initScrollReveal();

        // Load data
        const loaded = await loadData();

        if (loaded && samples.length > 0) {
            renderSampleList();
            animateHeroStats();

            // Auto-select first sample
            selectSample(samples[0].id);
        } else {
            showErrorState();
            // Still animate hero stats with defaults
            animateHeroStats();
        }

        console.log('[XAI Demo] Ready');
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
