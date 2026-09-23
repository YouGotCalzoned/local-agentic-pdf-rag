class AgentRuntime:
    """
    Holds resources associated with the currently loaded document.

    Tools can access this runtime without owning document loading
    or application state themselves.
    """

    def __init__(self):
        self.vector_store = None
        self.document_info = None
         # Evidence accumulated across search_document calls
        # during the current agent execution.
        self.retrieved_evidence = {}

    def reset_evidence(self):
        """
        Clear evidence accumulated during the previous agent run.
        """
        self.retrieved_evidence = {}

    def add_evidence(self, source_id, passage):
        """
        Store retrieved evidence by stable chunk ID.

        Using source_id as the key automatically deduplicates
        chunks retrieved by multiple searches.
        """
        self.retrieved_evidence[source_id] = passage

    def get_evidence(self):
        """
        Return all unique evidence accumulated so far.
        """
        return list(self.retrieved_evidence.values())

    def set_document(self, vector_store, document_info):
        self.vector_store = vector_store
        self.document_info = document_info

    def clear_document(self):
        self.vector_store = None
        self.document_info = None
        self.reset_evidence()

    def has_document(self):
        return self.vector_store is not None


agent_runtime = AgentRuntime()