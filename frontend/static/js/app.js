(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const config = {
    maxUploadMb: Number(app.dataset.maxUploadMb) || 25,
    topKDefault: Number(app.dataset.topKDefault) || 5,
    topKMax: Number(app.dataset.topKMax) || 20,
    allowedExtensions: (app.dataset.allowedExtensions || "pdf,txt")
      .split(",")
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean),
    embeddingModel: app.dataset.embeddingModel || "",
    version: app.dataset.appVersion || "1.0.0",
  };

  const byId = (id) => document.getElementById(id);
  const els = {
    healthChip: byId("healthChip"),
    healthDot: byId("healthDot"),
    healthText: byId("healthText"),
    statDocuments: byId("statDocuments"),
    statChunks: byId("statChunks"),
    statModel: byId("statModel"),
    searchForm: byId("searchForm"),
    searchInput: byId("searchInput"),
    searchButton: byId("searchButton"),
    searchSpinner: byId("searchSpinner"),
    searchBtnIcon: byId("searchBtnIcon"),
    searchBtnText: byId("searchBtnText"),
    topK: byId("topK"),
    exampleQueries: byId("exampleQueries"),
    resultsSection: byId("resultsSection"),
    resultsTitle: byId("resultsTitle"),
    resultsSubtitle: byId("resultsSubtitle"),
    resultsSummary: byId("resultsSummary"),
    searchLoading: byId("searchLoading"),
    resultsList: byId("resultsList"),
    searchEmpty: byId("searchEmpty"),
    searchEmptyTitle: byId("searchEmptyTitle"),
    searchEmptyText: byId("searchEmptyText"),
    searchEmptyActions: byId("searchEmptyActions"),
    dropzone: byId("dropzone"),
    fileInput: byId("fileInput"),
    dropzoneProgress: byId("dropzoneProgress"),
    dropzoneProgressText: byId("dropzoneProgressText"),
    uploadQueue: byId("uploadQueue"),
    uploadLimitBadge: byId("uploadLimitBadge"),
    documentList: byId("documentList"),
    documentLoading: byId("documentLoading"),
    documentsEmpty: byId("documentsEmpty"),
    documentCount: byId("documentCount"),
    libraryFooter: byId("libraryFooter"),
    libraryFooterText: byId("libraryFooterText"),
    indexWarning: byId("indexWarning"),
    indexWarningText: byId("indexWarningText"),
    refreshButton: byId("refreshButton"),
    rebuildButton: byId("rebuildButton"),
    emptyUploadButton: byId("emptyUploadButton"),
    deleteModalEl: byId("deleteModal"),
    deleteDocumentName: byId("deleteDocumentName"),
    confirmDeleteButton: byId("confirmDeleteButton"),
    deleteSpinner: byId("deleteSpinner"),
    rebuildModalEl: byId("rebuildModal"),
    confirmRebuildButton: byId("confirmRebuildButton"),
    rebuildSpinner: byId("rebuildSpinner"),
    toastStack: byId("toastStack"),
  };

  const state = {
    documents: [],
    health: null,
    searchBusy: false,
    uploading: 0,
    deleteTarget: null,
    rebuildBusy: false,
    lastQuery: "",
    lastTopK: config.topKDefault,
    resultsVisible: false,
    initialLoadDone: false,
  };

  const bootstrapModal = (element) => {
    if (!window.bootstrap || !window.bootstrap.Modal) return null;
    return window.bootstrap.Modal.getOrCreateInstance(element);
  };
  const deleteModal = bootstrapModal(els.deleteModalEl);
  const rebuildModal = bootstrapModal(els.rebuildModalEl);

  const ICONS = {
    pdf: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5 2.8h6.3L15 6.5v10.7H5V2.8Z" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/><path d="M11 2.8v3.9h4M7.2 10h5.5M7.2 12.4h3.8" stroke="currentColor" stroke-width="1.35" stroke-linecap="round"/></svg>`,
    txt: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5 2.8h6.3L15 6.5v10.7H5V2.8Z" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/><path d="M11 2.8v3.9h4M7.2 9.5h5.5M7.2 12h5.5M7.2 14.5h3.5" stroke="currentColor" stroke-width="1.35" stroke-linecap="round"/></svg>`,
    trash: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 5.7h12M7.5 5.7V4.2c0-.5.4-.9.9-.9h3.2c.5 0 .9.4.9.9v1.5m-8 0 .6 9.2c0 .5.4.9.9.9h4.8c.5 0 .9-.4.9-.9l.6-9.2" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    success: `<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m3.5 8.4 3 3 6-6.8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    error: `<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 4v5M8 11.5v.2" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.4"/></svg>`,
    info: `<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 7.2v4.3M8 4.7v.2" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.4"/></svg>`,
    source: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5 2.8h6.3L15 6.5v10.7H5V2.8Z" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/><path d="M11 2.8v3.9h4" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/></svg>`,
  };

  class ApiError extends Error {
    constructor(message, status = 0, code = "request_failed", details = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
      this.details = details;
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function extensionOf(name) {
    const parts = String(name || "").toLowerCase().split(".");
    return parts.length > 1 ? parts.pop() : "";
  }

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatDate(iso) {
    if (!iso) return "Unknown date";
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return "Unknown date";
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(parsed);
  }

  function pluralize(count, singular, plural = `${singular}s`) {
    return `${count.toLocaleString()} ${count === 1 ? singular : plural}`;
  }

  async function requestJson(path, options = {}) {
    let response;
    try {
      response = await fetch(path, {
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {}),
        },
      });
    } catch (error) {
      throw new ApiError(
        "Could not reach the server. Check that the Flask app is running.",
        0,
        "network_error"
      );
    }

    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }
    }

    if (!response.ok) {
      const apiError = payload && payload.error ? payload.error : {};
      throw new ApiError(
        apiError.message || `Request failed with status ${response.status}.`,
        response.status,
        apiError.code || "request_failed",
        apiError.details || null
      );
    }

    return payload ? payload.data : null;
  }

  function updateStats(health = null) {
    const docs = health ? health.database.documents : state.documents.length;
    const chunks = health
      ? health.database.chunks
      : state.documents.reduce((total, item) => total + (item.chunk_count || 0), 0);

    els.statDocuments.textContent = pluralize(docs, "document");
    els.statChunks.textContent = pluralize(chunks, "chunk");
    els.documentCount.textContent = String(docs);
    els.statDocuments.title = `${docs.toLocaleString()} uploaded documents`;
    els.statChunks.title = `${chunks.toLocaleString()} indexed text chunks`;

    const embedding = health ? health.embedding : null;
    let modelLabel = config.embeddingModel.split("/").pop() || "Embedding model";
    if (embedding) {
      if (embedding.backend === "hashing") {
        modelLabel = "Hashing fallback";
      } else {
        modelLabel = String(embedding.model || modelLabel).split("/").pop();
      }
      const details = [
        embedding.backend,
        embedding.model,
        embedding.dimension ? `${embedding.dimension} dimensions` : null,
        embedding.loaded ? "loaded" : "loads on first search",
      ].filter(Boolean);
      els.statModel.title = details.join(" · ");
    }
    els.statModel.textContent = modelLabel;
  }

  function setHealthChip(kind, text, title = "") {
    els.healthDot.classList.remove("is-ok", "is-warn", "is-error", "is-loading");
    els.healthDot.classList.add(`is-${kind}`);
    els.healthText.textContent = text;
    if (title) {
      els.healthChip.title = title;
    } else {
      els.healthChip.removeAttribute("title");
    }
  }

  function renderHealth(health) {
    state.health = health;
    updateStats(health);

    const index = health.index || {};
    const details = [
      index.backend ? `${String(index.backend).toUpperCase()} index` : null,
      `${(index.vectors || 0).toLocaleString()} vectors`,
      health.embedding ? health.embedding.model : null,
      health.database ? `${health.database.documents} documents` : null,
    ].filter(Boolean);

    if (health.status !== "ok") {
      setHealthChip("error", "Backend error", details.join(" · "));
    } else if (index.in_sync === false) {
      setHealthChip("warn", "Index needs rebuild", details.join(" · "));
    } else {
      setHealthChip("ok", "Index ready", details.join(" · "));
    }

    const warnings = Array.isArray(health.warnings) ? health.warnings : [];
    if (health.status === "ok" && (index.in_sync === false || warnings.length > 0)) {
      els.indexWarning.hidden = false;
      els.indexWarningText.textContent =
        warnings[0] || "The vector index is out of sync with your documents.";
    } else {
      els.indexWarning.hidden = true;
    }
  }

  async function loadHealth({ silent = false } = {}) {
    try {
      const health = await requestJson("/api/health");
      renderHealth(health);
      return health;
    } catch (error) {
      setHealthChip("error", "Backend offline", error.message);
      if (!silent) {
        showToast("Status check failed", error.message, "error");
      }
      return null;
    }
  }

  function createMetaSeparator() {
    return el("span", "doc-meta-dot", "•");
  }

  function renderDocuments() {
    els.documentList.replaceChildren();
    els.documentLoading.hidden = true;

    const hasDocuments = state.documents.length > 0;
    els.documentsEmpty.hidden = hasDocuments;
    els.libraryFooter.hidden = !hasDocuments;
    els.documentCount.textContent = String(state.documents.length);

    if (!hasDocuments) {
      updateStats(state.health);
      return;
    }

    els.libraryFooterText.textContent = `Stored locally · ${pluralize(
      state.documents.length,
      "source"
    )}`;

    const fragment = document.createDocumentFragment();
    state.documents.forEach((doc, index) => {
      const extension = extensionOf(doc.name);
      const row = el("div", "doc-row");
      row.style.animationDelay = `${Math.min(index, 6) * 25}ms`;

      const icon = el("div", `doc-file-icon is-${extension === "pdf" ? "pdf" : "txt"}`);
      icon.innerHTML = extension === "pdf" ? ICONS.pdf : ICONS.txt;
      icon.setAttribute("aria-hidden", "true");

      const main = el("div", "doc-main");
      const name = el("p", "doc-name", doc.name);
      name.title = doc.name;
      main.append(name);

      const meta = el("div", "doc-meta");
      if (doc.page_count > 0) {
        meta.append(el("span", "", pluralize(doc.page_count, "page")));
        meta.append(createMetaSeparator());
      }
      meta.append(el("span", "", pluralize(doc.chunk_count || 0, "chunk")));
      meta.append(createMetaSeparator());
      meta.append(el("span", "", formatBytes(doc.file_size)));
      meta.append(createMetaSeparator());
      meta.append(el("span", "", formatDate(doc.created_at)));
      meta.append(createMetaSeparator());
      const status = el("span", "doc-status", "Indexed");
      status.title = "Document metadata and vectors are stored locally";
      meta.append(status);
      main.append(meta);

      const deleteButton = el("button", "doc-delete");
      deleteButton.type = "button";
      deleteButton.innerHTML = ICONS.trash;
      deleteButton.setAttribute("aria-label", `Delete ${doc.name}`);
      deleteButton.title = "Delete document";
      deleteButton.addEventListener("click", () => openDeleteModal(doc));

      row.append(icon, main, deleteButton);
      fragment.append(row);
    });

    els.documentList.append(fragment);
    updateStats(state.health);
  }

  async function loadDocuments({ showLoading = true } = {}) {
    if (showLoading) {
      els.documentLoading.hidden = false;
      els.documentsEmpty.hidden = true;
    }
    try {
      const documents = await requestJson("/api/documents");
      state.documents = Array.isArray(documents) ? documents : [];
      renderDocuments();
      return state.documents;
    } catch (error) {
      els.documentLoading.hidden = true;
      if (state.documents.length === 0) {
        els.documentsEmpty.hidden = false;
      }
      showToast("Could not load documents", error.message, "error");
      return null;
    } finally {
      state.initialLoadDone = true;
    }
  }

  function populateTopK() {
    const max = Math.max(1, config.topKMax);
    const candidates = Array.from(
      new Set(
        [config.topKDefault, 3, 5, 8, 10, 15, 20, max].filter(
          (value) => Number.isInteger(value) && value >= 1 && value <= max
        )
      )
    ).sort((a, b) => a - b);

    els.topK.replaceChildren(
      ...candidates.map((value) => {
        const option = new Option(String(value), String(value));
        return option;
      })
    );
    const defaultChoice = candidates.includes(config.topKDefault)
      ? config.topKDefault
      : candidates[candidates.length - 1];
    els.topK.value = String(defaultChoice);
    state.lastTopK = Number(defaultChoice);
    els.uploadLimitBadge.textContent = `${config.allowedExtensions
      .map((extension) => extension.toUpperCase())
      .join(" · ")}`;
  }

  function currentTopK() {
    const value = Number(els.topK.value);
    return Number.isInteger(value) && value > 0 ? value : config.topKDefault;
  }

  function setSearchBusy(busy) {
    state.searchBusy = busy;
    els.searchButton.disabled = busy;
    els.searchSpinner.classList.toggle("d-none", !busy);
    els.searchBtnIcon.classList.toggle("d-none", busy);
    els.searchBtnText.textContent = busy ? "Searching" : "Search";
    els.resultsSection.setAttribute("aria-busy", String(busy));
  }

  function clearSearchEmptyActions() {
    els.searchEmptyActions.replaceChildren();
  }

  function addSearchEmptyAction(label, className, handler) {
    const button = el("button", `btn btn-sm ${className}`, label);
    button.type = "button";
    button.addEventListener("click", handler);
    els.searchEmptyActions.append(button);
  }

  function showSearchEmpty({
    title = "No matching passages",
    text = "Try rephrasing your question or uploading more documents.",
    actions = [],
  } = {}) {
    els.resultsList.replaceChildren();
    els.searchEmpty.hidden = false;
    els.searchEmptyTitle.textContent = title;
    els.searchEmptyText.textContent = text;
    clearSearchEmptyActions();
    actions.forEach(({ label, className, handler }) => {
      addSearchEmptyAction(label, className, handler);
    });
  }

  function focusUpload() {
    els.dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => els.fileInput.click(), 250);
  }

  function openRebuildModal() {
    els.confirmRebuildButton.disabled = false;
    els.rebuildSpinner.classList.add("d-none");
    if (rebuildModal) {
      rebuildModal.show();
    }
  }

  function scoreLabel(percent) {
    if (percent >= 50) return "High match";
    if (percent >= 30) return "Related";
    return "Low match";
  }

  function scoreColor(percent) {
    if (percent >= 50) return "#059669";
    if (percent >= 30) return "#d97706";
    return "#64748b";
  }

  function buildResultCard(result, index) {
    const score = Number.isFinite(Number(result.score)) ? Number(result.score) : 0;
    const percent = Math.round(Math.max(0, Math.min(1, score)) * 100);
    const extension = extensionOf(result.document_name);

    const article = el("article", "result-card");
    article.style.animationDelay = `${Math.min(index, 6) * 28}ms`;
    article.append(el("div", "result-rank", String(index + 1)));

    const content = el("div", "result-content");
    const header = el("div", "result-header");
    const source = el("div", "result-source");

    const sourceIcon = el("span", "result-source-icon");
    sourceIcon.innerHTML = extension === "pdf" ? ICONS.source : ICONS.txt;
    sourceIcon.setAttribute("aria-hidden", "true");
    const docName = el("span", "result-doc-name", result.document_name || "Unknown document");
    docName.title = result.document_name || "Unknown document";
    source.append(sourceIcon, docName);

    const pageNumber = Number(result.page_number);
    if (Number.isInteger(pageNumber) && pageNumber > 0) {
      const pageBadge = el("span", "meta-badge page-badge", `Page ${pageNumber}`);
      pageBadge.title = `Source page ${pageNumber}`;
      source.append(pageBadge);
    } else {
      const typeBadge = el(
        "span",
        "meta-badge txt-badge",
        extension ? extension.toUpperCase() : "TEXT"
      );
      typeBadge.title = "Plain text source (no page numbers)";
      source.append(typeBadge);
    }

    const scoreBlock = el("div", "score-block");
    const dial = el("div", "score-dial");
    dial.style.setProperty("--pct", String(percent));
    dial.style.setProperty("--score-color", scoreColor(percent));
    dial.title = `Cosine similarity: ${score.toFixed(4)}`;
    dial.setAttribute(
      "aria-label",
      `Similarity ${percent} percent, ${score.toFixed(3)} cosine score`
    );
    const inner = el("div", "score-dial-inner");
    inner.append(el("strong", "", `${percent}%`), el("span", "", "match"));
    dial.append(inner);

    const caption = el("div", "score-caption");
    caption.append(el("strong", "", scoreLabel(percent)), el("span", "", "cosine match"));
    scoreBlock.append(dial, caption);

    header.append(source, scoreBlock);

    const text = el("p", "result-text is-clamped", result.text || "");
    const footer = el("div", "result-footer");
    footer.append(
      el("span", "result-footer-meta", `Chunk ${Number(result.chunk_index) + 1}`),
      el("span", "result-footer-meta", `Similarity ${score.toFixed(4)}`)
    );

    if ((result.text || "").length > 420) {
      const toggle = el("button", "expand-button", "Show full passage");
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.addEventListener("click", () => {
        const expanded = text.classList.toggle("is-clamped") === false;
        toggle.textContent = expanded ? "Show less" : "Show full passage";
        toggle.setAttribute("aria-expanded", String(expanded));
      });
      footer.append(toggle);
    }

    content.append(header, text, footer);
    article.append(content);
    return article;
  }

  function renderResults(data) {
    const results = Array.isArray(data.results) ? data.results : [];
    state.resultsVisible = true;
    els.resultsSection.hidden = false;
    els.searchLoading.hidden = true;
    els.resultsSubtitle.textContent = `“${data.query || state.lastQuery}”`;
    els.resultsTitle.textContent = "Semantic matches";

    els.resultsSummary.replaceChildren();
    const strong = el("span", "summary-strong", String(results.length));
    els.resultsSummary.append(
      strong,
      document.createTextNode(
        ` ${results.length === 1 ? "passage" : "passages"} · ${
          data.duration_ms || 0
        } ms`
      )
    );

    if (results.length === 0) {
      const reason = data.reason || "no_matches";
      const actions = [];
      if (reason === "no_documents") {
        actions.push({
          label: "Upload a document",
          className: "btn-primary-soft",
          handler: focusUpload,
        });
      } else if (reason === "index_empty") {
        actions.push({
          label: "Rebuild index",
          className: "btn-primary",
          handler: openRebuildModal,
        });
        loadHealth({ silent: true });
      } else {
        actions.push({
          label: "Refine your question",
          className: "btn-primary-soft",
          handler: () => {
            els.searchInput.focus();
            els.searchInput.select();
          },
        });
        actions.push({
          label: "Add documents",
          className: "btn-light-soft",
          handler: focusUpload,
        });
      }
      showSearchEmpty({
        title:
          reason === "no_documents"
            ? "Your library is empty"
            : reason === "index_empty"
            ? "The index needs rebuilding"
            : "No matching passages",
        text: data.message || "Try rephrasing your question.",
        actions,
      });
      return;
    }

    els.searchEmpty.hidden = true;
    const fragment = document.createDocumentFragment();
    results.forEach((result, index) => {
      fragment.append(buildResultCard(result, index));
    });
    els.resultsList.replaceChildren(fragment);

    if (data.index && data.index.vectors !== undefined && !state.health) {
      updateStats(null);
    }
  }

  function showSearchError(error) {
    els.searchLoading.hidden = true;
    els.resultsSection.hidden = false;
    els.resultsSubtitle.textContent = "The search could not be completed.";
    els.resultsSummary.replaceChildren();
    showSearchEmpty({
      title: "Search failed",
      text: error.message,
      actions: [
        {
          label: "Try again",
          className: "btn-primary-soft",
          handler: () => performSearch(state.lastQuery, { scroll: false }),
        },
      ],
    });
    showToast("Search failed", error.message, "error", 7000);
    loadHealth({ silent: true });
  }

  async function performSearch(
    rawQuery,
    { scroll = true, topK = currentTopK() } = {}
  ) {
    if (state.searchBusy) return;
    const query = String(rawQuery || "").trim();
    if (!query) {
      showToast("Enter a question", "Describe what you are looking for.", "info");
      els.searchInput.focus();
      return;
    }

    state.lastQuery = query;
    state.lastTopK = topK;
    els.searchInput.value = query;
    if ([...els.topK.options].some((option) => Number(option.value) === topK)) {
      els.topK.value = String(topK);
    }

    setSearchBusy(true);
    state.resultsVisible = true;
    els.resultsSection.hidden = false;
    els.searchEmpty.hidden = true;
    els.resultsList.replaceChildren();
    els.searchLoading.hidden = false;
    els.resultsSummary.replaceChildren();
    els.resultsTitle.textContent = "Searching";
    els.resultsSubtitle.textContent = `“${query}”`;

    if (scroll) {
      window.setTimeout(() => {
        els.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 40);
    }

    try {
      const data = await requestJson("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, top_k: topK }),
      });
      renderResults(data);
    } catch (error) {
      showSearchError(error);
    } finally {
      setSearchBusy(false);
    }
  }

  function updateUploadBusy() {
    const busy = state.uploading > 0;
    els.dropzone.classList.toggle("is-busy", busy);
    els.dropzoneProgress.hidden = !busy;
    els.dropzoneProgressText.textContent = "Processing upload…";
    els.dropzone.setAttribute("aria-busy", String(busy));
  }

  function validateFile(file) {
    const extension = extensionOf(file.name);
    if (!config.allowedExtensions.includes(extension)) {
      return `Unsupported file type. Use ${config.allowedExtensions
        .map((value) => `.${value}`)
        .join(" or ")}.`;
    }
    if (file.size === 0) {
      return "This file is empty.";
    }
    if (file.size > config.maxUploadMb * 1024 * 1024) {
      return `This file is larger than the ${config.maxUploadMb} MB limit.`;
    }
    return null;
  }

  function createUploadItem(file) {
    const extension = extensionOf(file.name);
    const item = el("div", "upload-item");
    const top = el("div", "upload-item-top");
    const icon = el("div", `upload-file-icon is-${extension === "pdf" ? "pdf" : "txt"}`);
    icon.innerHTML = extension === "pdf" ? ICONS.pdf : ICONS.txt;
    icon.setAttribute("aria-hidden", "true");
    const name = el("span", "upload-name", file.name);
    name.title = file.name;
    const status = el("span", "upload-status", "Waiting");
    const bar = el("div", "upload-bar");
    const fill = el("div", "upload-bar-fill");
    bar.append(fill);
    const message = el("div", "upload-message");
    message.hidden = true;

    top.append(icon, name, status);
    item.append(top, bar, message);
    els.uploadQueue.append(item);

    return { item, status, fill, bar, message };
  }

  function setUploadItemError(refs, message) {
    refs.item.classList.add("is-error");
    refs.status.textContent = "Failed";
    refs.message.hidden = false;
    refs.message.textContent = message;
    refs.bar.hidden = true;
  }

  function uploadWithProgress(file, refs) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/documents/upload");
      request.setRequestHeader("Accept", "application/json");

      request.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.max(1, Math.min(99, Math.round((event.loaded / event.total) * 100)));
        refs.fill.style.width = `${percent}%`;
        refs.status.textContent = `${percent}%`;
      });

      request.upload.addEventListener("load", () => {
        refs.fill.style.width = "100%";
        refs.status.textContent = "Embedding…";
      });

      request.addEventListener("load", () => {
        let payload = null;
        try {
          payload = JSON.parse(request.responseText);
        } catch (error) {
          payload = null;
        }

        if (request.status >= 200 && request.status < 300 && payload && payload.data) {
          resolve(payload.data);
          return;
        }

        const detail = payload && payload.error ? payload.error : {};
        reject(
          new ApiError(
            detail.message || `Upload failed with status ${request.status}.`,
            request.status,
            detail.code || "upload_failed",
            detail.details || null
          )
        );
      });

      request.addEventListener("error", () => {
        reject(
          new ApiError(
            "Network error while uploading. Is the server still running?",
            0,
            "network_error"
          )
        );
      });

      request.addEventListener("abort", () => {
        reject(new ApiError("The upload was cancelled.", 0, "upload_aborted"));
      });

      const form = new FormData();
      form.append("file", file, file.name);
      request.send(form);
    });
  }

  async function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    if (state.uploading > 0) {
      showToast("Upload in progress", "Wait for the current files to finish.", "info");
      return;
    }

    let uploadedCount = 0;

    for (const file of files) {
      const refs = createUploadItem(file);
      const validationError = validateFile(file);
      if (validationError) {
        setUploadItemError(refs, validationError);
        continue;
      }

      state.uploading += 1;
      updateUploadBusy();
      refs.item.classList.remove("is-error", "is-success");
      refs.status.textContent = "Uploading…";

      try {
        const doc = await uploadWithProgress(file, refs);
        refs.item.classList.add("is-success");
        refs.fill.style.width = "100%";
        refs.status.textContent = "Indexed";
        refs.message.hidden = false;
        refs.message.textContent = `${pluralize(
          doc.chunk_count || 0,
          "chunk"
        )} embedded · ${formatBytes(doc.file_size)}`;
        uploadedCount += 1;
        showToast(
          "Document indexed",
          `“${doc.name}” is ready to search (${pluralize(
            doc.chunk_count || 0,
            "chunk"
          )}).`,
          "success"
        );
      } catch (error) {
        setUploadItemError(refs, error.message);
        showToast("Upload failed", `${file.name}: ${error.message}`, "error", 7000);
      } finally {
        state.uploading = Math.max(0, state.uploading - 1);
        updateUploadBusy();
      }
    }

    if (uploadedCount > 0) {
      await Promise.allSettled([
        loadDocuments({ showLoading: false }),
        loadHealth({ silent: true }),
      ]);
      if (state.lastQuery && state.resultsVisible) {
        performSearch(state.lastQuery, { scroll: false, topK: state.lastTopK });
      }
    }
  }

  function openDeleteModal(doc) {
    state.deleteTarget = doc;
    els.deleteDocumentName.textContent = `“${doc.name}”`;
    els.confirmDeleteButton.disabled = false;
    els.deleteSpinner.classList.add("d-none");
    if (deleteModal) {
      deleteModal.show();
    }
  }

  async function confirmDelete() {
    const target = state.deleteTarget;
    if (!target) return;

    els.confirmDeleteButton.disabled = true;
    els.deleteSpinner.classList.remove("d-none");

    try {
      const result = await requestJson(`/api/documents/${encodeURIComponent(target.id)}`, {
        method: "DELETE",
      });
      state.deleteTarget = null;
      if (deleteModal) deleteModal.hide();

      showToast(
        "Document deleted",
        `“${target.name}” and ${pluralize(
          result && result.vectors_removed !== undefined
            ? result.vectors_removed
            : target.chunk_count || 0,
          "vector"
        )} were removed.`,
        "success"
      );

      await Promise.allSettled([
        loadDocuments({ showLoading: false }),
        loadHealth({ silent: true }),
      ]);

      if (state.lastQuery && state.resultsVisible) {
        performSearch(state.lastQuery, { scroll: false, topK: state.lastTopK });
      }
    } catch (error) {
      const message =
        error.status === 404
          ? "This document was already deleted. The library will be refreshed."
          : error.message;
      showToast("Delete failed", message, "error", 7000);
      if (error.status === 404) {
        if (deleteModal) deleteModal.hide();
        state.deleteTarget = null;
        await Promise.allSettled([
          loadDocuments({ showLoading: false }),
          loadHealth({ silent: true }),
        ]);
      }
    } finally {
      els.confirmDeleteButton.disabled = false;
      els.deleteSpinner.classList.add("d-none");
    }
  }

  async function confirmRebuild() {
    if (state.rebuildBusy) return;
    state.rebuildBusy = true;
    els.confirmRebuildButton.disabled = true;
    els.rebuildSpinner.classList.remove("d-none");

    try {
      const result = await requestJson("/api/index/rebuild", { method: "POST" });
      if (rebuildModal) rebuildModal.hide();
      const seconds = ((result.duration_ms || 0) / 1000).toFixed(1);
      showToast(
        "Index rebuilt",
        `${pluralize(result.chunks || 0, "chunk")} re-embedded in ${seconds}s using ${
          result.backend ? String(result.backend).toUpperCase() : "the vector store"
        }.`,
        "success",
        6000
      );
      await Promise.allSettled([
        loadDocuments({ showLoading: false }),
        loadHealth({ silent: true }),
      ]);
      if (state.lastQuery && state.resultsVisible) {
        performSearch(state.lastQuery, { scroll: false, topK: state.lastTopK });
      }
    } catch (error) {
      showToast("Rebuild failed", error.message, "error", 8000);
      loadHealth({ silent: true });
    } finally {
      state.rebuildBusy = false;
      els.confirmRebuildButton.disabled = false;
      els.rebuildSpinner.classList.add("d-none");
    }
  }

  function showToast(title, message, type = "info", timeout = 5000) {
    while (els.toastStack.children.length >= 4) {
      els.toastStack.firstElementChild.remove();
    }

    const toast = el("div", "app-toast");
    toast.setAttribute("role", "status");
    const icon = el("div", `toast-icon ${type}`);
    icon.innerHTML = ICONS[type] || ICONS.info;
    icon.setAttribute("aria-hidden", "true");
    const copy = el("div", "toast-copy");
    copy.append(el("div", "toast-title", title), el("div", "toast-message", message));
    const close = el("button", "toast-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss notification");
    close.addEventListener("click", () => dismissToast(toast));
    toast.append(icon, copy, close);
    els.toastStack.append(toast);

    const timer = window.setTimeout(() => dismissToast(toast), timeout);
    toast.dataset.timer = String(timer);
  }

  function dismissToast(toast) {
    if (!toast || !toast.isConnected || toast.classList.contains("is-leaving")) return;
    const timer = Number(toast.dataset.timer);
    if (timer) window.clearTimeout(timer);
    toast.classList.add("is-leaving");
    window.setTimeout(() => toast.remove(), 210);
  }

  function bindEvents() {
    els.searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      performSearch(els.searchInput.value, { topK: currentTopK() });
    });

    els.topK.addEventListener("change", () => {
      state.lastTopK = currentTopK();
      if (state.lastQuery && state.resultsVisible && !state.searchBusy) {
        performSearch(state.lastQuery, { scroll: false, topK: state.lastTopK });
      }
    });

    els.exampleQueries.addEventListener("click", (event) => {
      const button = event.target.closest("[data-query]");
      if (!button) return;
      const query = button.dataset.query || "";
      els.searchInput.value = query;
      performSearch(query, { topK: currentTopK() });
    });

    els.dropzone.addEventListener("click", () => {
      if (state.uploading > 0) return;
      els.fileInput.click();
    });

    els.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (state.uploading === 0) els.fileInput.click();
      }
    });

    els.fileInput.addEventListener("change", () => {
      const files = Array.from(els.fileInput.files || []);
      els.fileInput.value = "";
      handleFiles(files);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      els.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (state.uploading === 0) els.dropzone.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      els.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        els.dropzone.classList.remove("is-dragging");
      });
    });

    els.dropzone.addEventListener("drop", (event) => {
      const files = event.dataTransfer ? event.dataTransfer.files : null;
      if (files && files.length > 0) handleFiles(files);
    });

    els.emptyUploadButton.addEventListener("click", focusUpload);

    els.refreshButton.addEventListener("click", async () => {
      if (els.refreshButton.classList.contains("is-spinning")) return;
      els.refreshButton.classList.add("is-spinning");
      await Promise.allSettled([
        loadDocuments({ showLoading: true }),
        loadHealth({ silent: true }),
      ]);
      els.refreshButton.classList.remove("is-spinning");
    });

    els.rebuildButton.addEventListener("click", openRebuildModal);
    document.querySelectorAll("[data-open-rebuild]").forEach((button) => {
      button.addEventListener("click", openRebuildModal);
    });

    els.confirmDeleteButton.addEventListener("click", confirmDelete);
    els.confirmRebuildButton.addEventListener("click", confirmRebuild);

    els.deleteModalEl.addEventListener("hidden.bs.modal", () => {
      state.deleteTarget = null;
    });

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) loadHealth({ silent: true });
    });

    window.addEventListener("online", () => loadHealth({ silent: true }));
    window.addEventListener("offline", () =>
      setHealthChip("error", "Backend offline", "The browser is offline.")
    );
  }

  function init() {
    populateTopK();
    bindEvents();
    els.uploadLimitBadge.textContent = `${config.allowedExtensions
      .map((extension) => extension.toUpperCase())
      .join(" · ")} · up to ${config.maxUploadMb} MB`;
    updateUploadBusy();
    loadDocuments({ showLoading: true });
    loadHealth({ silent: true });

    window.setInterval(() => {
      if (!document.hidden) loadHealth({ silent: true });
    }, 30000);
  }

  init();
})();
