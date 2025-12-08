"""
Sanskrit Inference Tokenizer - Minimal loader-only tokenizer.

This module provides the SanskritTokenizer class optimized for inference,
stripping out all training-side vocabulary generation logic and constant lists.
"""

import re
from typing import List, Tuple, Optional, Dict
import unicodedata

class SanskritTokenizer:
    """
    Custom tokenizer for Sanskrit (Devanagari) text.
    
    Decomposes text into phonemes (Varnas) and maps them to IDs based on a provided vocabulary.
    """
    
    # Devanagari Unicode ranges (needed for decomposition)
    DEVANAGARI_VOWELS = 'अआइईउऊऋॠऌॡएऐओऔ'
    DEVANAGARI_CONSONANTS = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
    
    # Vowel signs (matras)
    VOWEL_SIGNS = {
        'ा': 'आ',  # aa
        'ि': 'इ',  # i
        'ी': 'ई',  # ii
        'ु': 'उ',  # u
        'ू': 'ऊ',  # uu
        'ृ': 'ऋ',  # ri
        'ॄ': 'ॠ',  # rii
        'ॢ': 'ऌ',  # li
        'ॣ': 'ॡ',  # lii
        'े': 'ए',  # e
        'ै': 'ऐ',  # ai
        'ो': 'ओ',  # o
        'ौ': 'औ',  # au
    }
    
    # Special characters
    HALANT = '्'  # Virama/halant
    ANUSVARA = 'ं'  # Anusvara
    VISARGA = 'ः'  # Visarga
    CHANDRABINDU = 'ँ'  # Chandrabindu
    
    # Punctuation and symbols
    PUNCTUATION = ['|', '||', '।', '॥', '.', '?', '!', ',', ';', '-', '(', ')', '"', "'", ':', '=', '+', '/', '\\', '*', '%', '&', '$', '#', '@', '[', ']', '{', '}', '<', '>', '`', '~', '_']
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    SPACE_TOKEN = '<SPACE>'
    NEWLINE_TOKEN = '<NL>'
    NEWLINE_MARKER = '<NEWLINE_MARKER>'
    
    def __init__(self, max_length: int = 512, vocab_list: Optional[List[str]] = None, token_to_id: Optional[Dict[str, int]] = None):
        """
        Initialize the Sanskrit tokenizer.
        
        Args:
            max_length: Maximum sequence length
            vocab_list: Optional list of vocabulary tokens
            token_to_id: Optional mapping of token to ID
        
        Raises:
            ValueError: If neither vocab_list nor token_to_id is provided.
        """
        self.max_length = max_length
        
        if vocab_list is not None:
            self.vocab = vocab_list
            if token_to_id is not None:
                self.token_to_id = token_to_id
            else:
                self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        elif token_to_id is not None:
            self.token_to_id = token_to_id
            # Reconstruct vocab list from dict (assuming it's complete)
            self.vocab = [None] * len(token_to_id)
            for token, idx in token_to_id.items():
                if idx < len(self.vocab):
                    self.vocab[idx] = token
            # Filter None in case of gaps (shouldn't happen in standard vocab)
            self.vocab = [v for v in self.vocab if v is not None]
        else:
            # We strictly fail if no vocab is provided during inference init
            # The secret building logic is removed.
            raise ValueError("SanskritTokenizer requires a vocabulary (vocab_list or token_to_id) during initialization.")
            
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Sanskrit text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize Unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # Replace sequences of non-newline whitespace with a single space
        # [^\S\n] matches any whitespace character except newline
        text = re.sub(r'[^\S\n]+', ' ', text)
        
        return text.strip()
    
    def decompose_word(self, word: str) -> List[str]:
        """
        Decompose a Sanskrit word into phonemes.
        
        Args:
            word: Sanskrit word in Devanagari
            
        Returns:
            List of phonemes
        """
        phonemes = []
        
        if word == self.NEWLINE_MARKER:
            return [self.NEWLINE_TOKEN]
            
        i = 0
        
        while i < len(word):
            char = word[i]
            
            # Check for double danda
            if i + 1 < len(word) and char == '|' and word[i + 1] == '|':
                phonemes.append('||')
                i += 2
                continue
            
            # Check for newline
            if char == '\n':
                phonemes.append(self.NEWLINE_TOKEN)
                i += 1
                continue
            
            # Check for punctuation
            if char in self.PUNCTUATION:
                phonemes.append(char)
                i += 1
                continue
            
            # Check for consonant clusters (look ahead)
            if i + 1 < len(word) and word[i + 1] == self.HALANT:
                # Consonant with halant
                if i + 2 < len(word):
                    # Check if followed by another consonant (conjunct)
                    next_char = word[i + 2]
                    if next_char in self.DEVANAGARI_CONSONANTS:
                        # This is a conjunct, add halant consonant
                        phonemes.append(char + self.HALANT)
                        i += 2
                        continue
                    else:
                        # Halant at end or before non-consonant
                        phonemes.append(char + self.HALANT)
                        i += 2
                        continue
                else:
                    # Halant at end of word
                    phonemes.append(char + self.HALANT)
                    i += 2
                    continue
            
            # Check for consonant with matra
            if char in self.DEVANAGARI_CONSONANTS and i + 1 < len(word):
                next_char = word[i + 1]
                if next_char in self.VOWEL_SIGNS:
                    # Consonant + matra
                    phonemes.append(char + next_char)
                    i += 2
                    continue
            
            # Check for standalone vowel
            if char in self.DEVANAGARI_VOWELS:
                phonemes.append(char)
                i += 1
                continue
            
            # Check for standalone consonant (has implicit 'a')
            if char in self.DEVANAGARI_CONSONANTS:
                phonemes.append(char)
                i += 1
                continue
            
            # Check for special characters
            if char in [self.ANUSVARA, self.VISARGA, self.CHANDRABINDU]:
                # Attach to previous phoneme if exists
                if phonemes:
                    phonemes[-1] += char
                else:
                    phonemes.append(char)
                i += 1
                continue
            
            # Skip spaces and unknown characters
            i += 1
        
        return phonemes
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum length (uses self.max_length if None)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_length
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Preserve newlines by replacing with marker
        text = text.replace('\n', f' {self.NEWLINE_MARKER} ')
        
        # Split into words and decompose
        words = text.split()
        phonemes = []
        
        for i, word in enumerate(words):
            word_phonemes = self.decompose_word(word)
            phonemes.extend(word_phonemes)
            # Add space token between words (but not after the last word)
            if i < len(words) - 1:
                phonemes.append(self.SPACE_TOKEN)
        
        # Add special tokens
        if add_special_tokens:
            phonemes = [self.BOS_TOKEN] + phonemes + [self.EOS_TOKEN]
        
        # Truncate if needed
        if truncation and len(phonemes) > max_length:
            phonemes = phonemes[:max_length]
            if add_special_tokens:
                phonemes[-1] = self.EOS_TOKEN
        
        # Convert to IDs
        input_ids = [
            self.token_to_id.get(phoneme, self.token_to_id.get(self.UNK_TOKEN, 0))
            for phoneme in phonemes
        ]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            # Get pad token ID safely
            pad_id = self.token_to_id.get(self.PAD_TOKEN, 0)
            input_ids.extend([pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        # Ensure input_ids are integers
        input_ids = [int(x) for x in input_ids]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            
            if skip_special_tokens:
                if token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                    continue
            
            tokens.append(token)
        
        # Combine tokens back into text, replacing SPACE tokens with actual spaces
        text_parts = []
        current_word = []
        
        for token in tokens:
            if token == self.SPACE_TOKEN:
                if current_word:
                    text_parts.append(''.join(current_word))
                    current_word = []
            elif token == self.NEWLINE_TOKEN:
                if current_word:
                    text_parts.append(''.join(current_word))
                    current_word = []
                text_parts.append('\n')
            else:
                current_word.append(token)
        
        # Add final word
        if current_word:
            text_parts.append(''.join(current_word))
        
        text = ' '.join(text_parts)
        
        # Clean up spaces around newlines
        text = text.replace(' \n ', '\n').replace(' \n', '\n').replace('\n ', '\n')
        
        return text
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
