import abc
from typing import List, Protocol


class DocLayoutPolicy(Protocol):
    """
    Policy describing how each document is laid out in the packed sequence.

    Conceptually, every document in a batch contributes three segments:

        [prefix(doc_id)] + [body(doc_id)] + [suffix(doc_id)]

    The pack sampler is responsible for budgeting the *total* number of tokens
    per batch (including prefix and suffix), but it only ever truncates the
    body segment. Implementations of this protocol may use graph metadata,
    tokenizers, and caching internally, but expose only simple length and
    token-accessors here.
    """

    def prefix_length(self, doc_id: int) -> int:
        """Number of prefix tokens emitted before the body for this doc."""

    def suffix_length(self, doc_id: int) -> int:
        """Number of suffix tokens emitted after the body for this doc."""

    def prefix_tokens(self, doc_id: int) -> List[int]:
        """
        Token ids for the prefix; to be consumed later in the collate layer.

        The pack sampler does not inspect these tokens; it only reasons about
        lengths. The collate function will use these tokens when materialising
        the final packed tensor.
        """

    def suffix_tokens(self, doc_id: int) -> List[int]:
        """
        Token ids for the suffix; to be consumed later in the collate layer.

        As with ``prefix_tokens``, the sampler only needs lengths; callers in
        the collate layer will use these tokens when building the batch.
        """


class NullLayoutPolicy(DocLayoutPolicy):
    """
    Trivial layout policy that adds no decoration around document bodies.

    Under this policy, each document contributes exactly its body tokens to the
    batch and no additional prefix or suffix tokens. This preserves the current
    semantics of ``PackBatchSampler`` where ``effective_len`` equals the total
    number of tokens contributed by the document.
    """

    def prefix_length(self, doc_id: int) -> int:  # noqa: ARG002
        return 0

    def suffix_length(self, doc_id: int) -> int:  # noqa: ARG002
        return 0

    def prefix_tokens(self, doc_id: int) -> List[int]:  # noqa: ARG002
        return []

    def suffix_tokens(self, doc_id: int) -> List[int]:  # noqa: ARG002
        return []


class BOSEOSLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that adds a beginning-of-sequence (BOS) token as the prefix
    and an end-of-sequence (EOS) token as the suffix for each document.
    """

    def __init__(self, bos_token_id: int, eos_token_id: int):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prefix_length(self, doc_id: int) -> int:  # noqa: ARG002
        return 1

    def suffix_length(self, doc_id: int) -> int:  # noqa: ARG002
        return 1

    def prefix_tokens(self, doc_id: int) -> List[int]:  # noqa: ARG002
        return [self.bos_token_id]

    def suffix_tokens(self, doc_id: int) -> List[int]:  # noqa: ARG002
        return [self.eos_token_id]
