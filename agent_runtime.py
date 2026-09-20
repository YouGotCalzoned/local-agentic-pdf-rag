class AgentRuntime:
    """
    Holds resources associated with the currently loaded document.

    Tools can access this runtime without owning document loading
    or application state themselves.
    """

    def __init__(self):
        self.vector_store = None
        self.document_info = None

    def set_document(self, vector_store, document_info):
        self.vector_store = vector_store
        self.document_info = document_info

    def clear_document(self):
        self.vector_store = None
        self.document_info = None

    def has_document(self):
        return self.vector_store is not None


agent_runtime = AgentRuntime()