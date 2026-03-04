(function initOfflineInferenceCore(global) {
  class InferenceBackend {
    constructor(name) {
      this.name = String(name || "backend");
      this.ready = false;
      this.capabilities = {
        bio: false,
        word: false,
      };
      this.lastError = "";
    }

    async init() {
      this.ready = true;
      return {
        ok: true,
        backend: this.name,
        message: "initialized",
      };
    }

    async dispose() {
      this.ready = false;
    }

    isReady() {
      return Boolean(this.ready);
    }

    async predictBio() {
      throw new Error("predictBio is not implemented in this backend");
    }

    async predictWord() {
      throw new Error("predictWord is not implemented in this backend");
    }
  }

  const namespace = global.OfflineInference || {};
  namespace.InferenceBackend = InferenceBackend;
  global.OfflineInference = namespace;
})(window);
