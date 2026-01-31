"""
Protocol for detecting links in tokenized sequences.

This abstraction allows different link detection strategies for different
content types (markdown, Python imports, LaTeX citations, etc.) without
hardcoding token patterns in the mask creator.
"""
from typing import Protocol, List, Callable, NamedTuple
import torch


class LinkInfo(NamedTuple):
    """Information about a detected link in the token sequence."""
    link_start_pos: int      # Position where link pattern starts (e.g., '[' in markdown)
    link_mid_pos: int        # Position of separator (e.g., '](' in markdown, 'import' in Python)
    link_end_pos: int        # Position where link pattern ends (e.g., ')' in markdown)
    target_start: int        # Start of target identifier tokens
    target_end: int          # End of target identifier tokens (exclusive)


class TokenizedLinkDetector(Protocol):
    """
    Detects links in tokenized sequences.
    
    Different implementations handle different content types:
    - MarkdownLinkDetector: Finds [text](target) patterns, decodes target from tokens
    - PythonImportDetector: Finds import positions, uses doc_spans.outgoing_titles for targets
    - LaTeXCitationDetector: Finds \cite{target} patterns (future)
    """
    
    uses_outgoing_titles: bool
    """If True, use doc_spans.outgoing_titles instead of decoding targets from tokens."""
    
    def detect_links(
        self,
        input_ids: torch.Tensor,
        tokenizer_decode_fn: Callable[[List[int]], str]
    ) -> List[LinkInfo]:
        """
        Detect all link patterns in the token sequence.
        
        Args:
            input_ids: 1D tensor of token IDs [seq_len]
            tokenizer_decode_fn: Function to decode token IDs to text
        
        Returns:
            List of LinkInfo objects. 
            - If uses_outgoing_titles=False: target_start/target_end contain actual target tokens
            - If uses_outgoing_titles=True: target_start/target_end are dummy (0, 0)
        """
        ...
