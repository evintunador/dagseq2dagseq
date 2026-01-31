"""
Concrete implementations of TokenizedLinkDetector for different content types.
"""
from typing import List, Callable, Optional
import torch
import logging

from .link_detector_protocol import LinkInfo, TokenizedLinkDetector

logger = logging.getLogger(__name__)


class MarkdownLinkDetector(TokenizedLinkDetector):
    """
    Detects markdown links: [text](target)
    
    Looks for token sequences matching:
    - '[' token (start)
    - Any text
    - '](' token (separator)
    - Target tokens
    - ')' token (end)
    
    Target is decoded from tokens and matched directly to clean_title.
    """
    
    uses_outgoing_titles = False
    
    def __init__(
        self,
        link_start_token_ids: Optional[List[int]] = None,
        link_mid_token_id: int = 16151,   # '](' in GPT-2
        link_end_token_id: int = 8,       # ')' in GPT-2
    ):
        """
        Args:
            link_start_token_ids: Token IDs for '[' (default: [58, 685] for GPT-2)
            link_mid_token_id: Token ID for '](' 
            link_end_token_id: Token ID for ')'
        """
        if link_start_token_ids is None:
            link_start_token_ids = [58, 685]  # '[' and ' ['
        
        self.link_start_token_ids = set(link_start_token_ids)
        self.link_mid_token_id = link_mid_token_id
        self.link_end_token_id = link_end_token_id
        
        logger.info(
            f"Initialized MarkdownLinkDetector with token IDs: "
            f"'[' = {link_start_token_ids}, '](' = {link_mid_token_id}, ')' = {link_end_token_id}"
        )
    
    def detect_links(
        self,
        input_ids: torch.Tensor,
        tokenizer_decode_fn: Callable[[List[int]], str]
    ) -> List[LinkInfo]:
        """
        Detect markdown link patterns [text](target) in token sequence.
        """
        links = []
        seq_len = input_ids.shape[0]
        
        # Find all positions of '](' token
        link_mid_positions = (input_ids == self.link_mid_token_id).nonzero(as_tuple=True)[0]
        
        for mid_pos in link_mid_positions:
            mid_pos = mid_pos.item()
            
            # Search backwards for '[' token
            link_start_pos = None
            for i in range(mid_pos - 1, max(-1, mid_pos - 101), -1):
                if input_ids[i].item() in self.link_start_token_ids:
                    link_start_pos = i
                    break
            
            if link_start_pos is None:
                continue
            
            # Search forwards for ')' token
            link_end_pos = None
            for i in range(mid_pos + 1, min(mid_pos + 101, seq_len)):
                if input_ids[i] == self.link_end_token_id:
                    link_end_pos = i
                    break
            
            if link_end_pos is None:
                continue
            
            # Calculate target span (between '](' and ')')
            target_start = mid_pos + 1
            target_end = link_end_pos
            
            if target_start >= target_end:
                continue
            
            links.append(LinkInfo(
                link_start_pos=link_start_pos,
                link_mid_pos=mid_pos,
                link_end_pos=link_end_pos,
                target_start=target_start,
                target_end=target_end
            ))
        
        logger.info(f"Found {len(links)} markdown links in batch")
        return links


class PythonImportDetector(TokenizedLinkDetector):
    """
    Detects Python import statement positions in tokenized code.
    
    For Python:
    - Finds WHERE import statements appear (positions)
    - Does NOT attempt to resolve module paths to file paths
    - Resolution was already done by graph builder (handles relative imports, __init__.py, etc.)
    - Uses doc_spans.outgoing_titles (from graph) for actual targets
    
    This maintains proper abstraction: graph does complex resolution once,
    mask creator just uses the results.
    """
    
    uses_outgoing_titles = True
    
    def __init__(
        self,
        import_token_id: int = 1330,     # 'import' in GPT-2
        from_token_id: int = 6738,       # 'from' in GPT-2
        newline_token_id: int = 198,     # '\n' in GPT-2
        dot_token_id: int = 13,          # '.' in GPT-2
    ):
        """
        Args:
            import_token_id: Token ID for 'import'
            from_token_id: Token ID for 'from'
            newline_token_id: Token ID for newline
            dot_token_id: Token ID for '.'
        """
        self.import_token_id = import_token_id
        self.from_token_id = from_token_id
        self.newline_token_id = newline_token_id
        self.dot_token_id = dot_token_id
        
        logger.info(
            f"Initialized PythonImportDetector with token IDs: "
            f"'import' = {import_token_id}, 'from' = {from_token_id}"
        )
    
    def detect_links(
        self,
        input_ids: torch.Tensor,
        tokenizer_decode_fn: Callable[[List[int]], str]
    ) -> List[LinkInfo]:
        """
        Detect import statement positions.
        
        Returns LinkInfo with just positions, not targets.
        Targets come from doc_spans.outgoing_titles (graph has already resolved them).
        """
        links = []
        
        # Find all 'from' and 'import' tokens
        from_positions = (input_ids == self.from_token_id).nonzero(as_tuple=True)[0]
        import_positions = (input_ids == self.import_token_id).nonzero(as_tuple=True)[0]
        
        # Record each import position
        # target_start=0, target_end=0 signals to use doc_spans.outgoing_titles
        for from_pos in from_positions:
            links.append(LinkInfo(
                link_start_pos=from_pos.item(),
                link_mid_pos=from_pos.item(),
                link_end_pos=from_pos.item(),
                target_start=0,
                target_end=0
            ))
        
        for import_pos in import_positions:
            # Skip if part of 'from...import'
            pos = import_pos.item()
            if any(from_pos < pos < from_pos + 30 for from_pos in from_positions):
                continue
            
            links.append(LinkInfo(
                link_start_pos=pos,
                link_mid_pos=pos,
                link_end_pos=pos,
                target_start=0,
                target_end=0
            ))
        
        logger.info(f"Found {len(links)} Python import positions in batch")
        return links
