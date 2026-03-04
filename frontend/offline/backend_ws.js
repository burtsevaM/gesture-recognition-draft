(function initBackendWsAdapter(global) {
  const namespace = global.OfflineInference || {};
  const Base = namespace.InferenceBackend || class {};

  class BackendWsInferenceBackend extends Base {
    constructor() {
      super("backend");
      this.capabilities = { bio: true, word: true };
    }

    async init() {
      this.ready = true;
      this.lastError = "";
      return {
        ok: true,
        backend: "backend",
        message: "WebSocket inference is active",
      };
    }

    async predictBio() {
      throw new Error("predictBio is handled by backend WebSocket stream");
    }

    async predictWord() {
      throw new Error("predictWord is handled by backend WebSocket stream");
    }
  }

  namespace.BackendWsInferenceBackend = BackendWsInferenceBackend;
  global.OfflineInference = namespace;
})(window);
