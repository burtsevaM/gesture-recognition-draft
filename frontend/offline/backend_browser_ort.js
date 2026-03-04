(function initBrowserOrtAdapter(global) {
  const namespace = global.OfflineInference || {};
  const Base = namespace.InferenceBackend || class {};
  const ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

  let ortLoadPromise = null;

  function injectScript(url) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[data-ort-url="${url}"]`);
      if (existing) {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener("error", () => reject(new Error("failed to load onnxruntime-web")), { once: true });
        return;
      }

      const script = document.createElement("script");
      script.src = url;
      script.async = true;
      script.dataset.ortUrl = url;
      script.addEventListener("load", () => resolve(), { once: true });
      script.addEventListener("error", () => reject(new Error("failed to load onnxruntime-web")), { once: true });
      document.head.appendChild(script);
    });
  }

  async function loadOrtGlobal() {
    if (global.ort) {
      return global.ort;
    }
    if (!ortLoadPromise) {
      ortLoadPromise = injectScript(ORT_CDN_URL).then(() => {
        if (!global.ort) {
          throw new Error("onnxruntime-web loaded but window.ort is missing");
        }
        return global.ort;
      });
    }
    return ortLoadPromise;
  }

  class BrowserOrtInferenceBackend extends Base {
    constructor() {
      super("browser");
      this.capabilities = { bio: true, word: true };
      this.ort = null;
      this.wordSession = null;
      this.bioSession = null;
      this.manifest = null;
      this.options = {};
    }

    async init(options = {}) {
      this.options = { ...options };
      this.lastError = "";
      this.ready = false;

      try {
        this.ort = await loadOrtGlobal();
      } catch (err) {
        this.lastError = `ORT init failed: ${err?.message || err}`;
        return { ok: false, backend: "browser", reason: this.lastError };
      }

      const executionProviders = Array.isArray(options.executionProviders) && options.executionProviders.length
        ? options.executionProviders
        : ["wasm"];

      if (this.ort?.env?.wasm) {
        this.ort.env.wasm.numThreads = Math.max(1, Number(options.wasmThreads || 1));
      }

      const wordModelUrl = String(options.wordModelUrl || "");
      const bioModelUrl = String(options.bioModelUrl || "");
      const manifestUrl = String(options.manifestUrl || "");

      if (manifestUrl) {
        try {
          const response = await fetch(manifestUrl);
          if (response.ok) {
            this.manifest = await response.json();
          }
        } catch (err) {
          // manifest optional
        }
      }

      if (!wordModelUrl || !bioModelUrl) {
        this.lastError = "model URLs are not configured";
        return { ok: false, backend: "browser", reason: this.lastError };
      }

      try {
        this.wordSession = await this.ort.InferenceSession.create(wordModelUrl, { executionProviders });
        this.bioSession = await this.ort.InferenceSession.create(bioModelUrl, { executionProviders });
        this.ready = true;
        return {
          ok: true,
          backend: "browser",
          message: `ORT sessions loaded (${executionProviders.join(",")})`,
        };
      } catch (err) {
        this.lastError = `model load failed: ${err?.message || err}`;
        this.wordSession = null;
        this.bioSession = null;
        this.ready = false;
        return { ok: false, backend: "browser", reason: this.lastError };
      }
    }

    async dispose() {
      this.wordSession = null;
      this.bioSession = null;
      this.ready = false;
    }

    async predictBio() {
      if (!this.ready || !this.bioSession) {
        throw new Error("browser BIO backend is not ready");
      }
      throw new Error("browser predictBio scaffolding only: runtime inference is not wired yet");
    }

    async predictWord() {
      if (!this.ready || !this.wordSession) {
        throw new Error("browser word backend is not ready");
      }
      throw new Error("browser predictWord scaffolding only: runtime inference is not wired yet");
    }
  }

  namespace.BrowserOrtInferenceBackend = BrowserOrtInferenceBackend;
  global.OfflineInference = namespace;
})(window);
