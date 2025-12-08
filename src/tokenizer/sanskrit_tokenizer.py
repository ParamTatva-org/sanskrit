"""
Sanskrit Tokenizer - Phoneme-based tokenization for Devanagari script.

This module provides custom tokenization that decomposes Sanskrit text into
phonemes (Varnas) based on the Paramtatva graph structure.
"""

import re
from typing import List, Tuple, Optional, Dict
import unicodedata




class SanskritTokenizer:
    """
    Custom tokenizer for Sanskrit (Devanagari) text.
    
    Decomposes text into phonemes (Varnas) rather than using statistical tokenization.
    """
    
    # Devanagari Unicode ranges
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
            vocab_list: Optional list of vocabulary tokens (if provided, skips _build_vocabulary)
            token_to_id: Optional mapping of token to ID (if provided, uses this instead of building)
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
            # Build vocabulary using internal logic (contains secrets)
            self.vocab = self._build_vocabulary()
            self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
            
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def _build_vocabulary(self) -> List[str]:
        """
        Build comprehensive vocabulary including all types of multi-phoneme combinations.
        
        Dictionary 2 (Compound): Contains all sound combinations with 2 or more phonemes,
        including conjuncts, clusters, and common multi-phoneme patterns.
        """
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SPACE_TOKEN, self.NEWLINE_TOKEN]
        
        # Add punctuation
        vocab.extend(self.PUNCTUATION)
        
        # Dictionary 1 (Core): Single phonemes from 14 Maheshwara Sutras
        # Add all vowels
        for vowel in self.DEVANAGARI_VOWELS:
            vocab.append(vowel)
        
        # Add all consonants (with implicit 'a')
        for consonant in self.DEVANAGARI_CONSONANTS:
            vocab.append(consonant)
        
        # Add all consonants with halant (pure consonant)
        for consonant in self.DEVANAGARI_CONSONANTS:
            vocab.append(consonant + self.HALANT)
        
        # Add consonant + vowel combinations
        for consonant in self.DEVANAGARI_CONSONANTS:
            for matra, vowel in self.VOWEL_SIGNS.items():
                vocab.append(consonant + matra)
        
        # Add special characters
        vocab.extend([self.ANUSVARA, self.VISARGA, self.CHANDRABINDU])
        
        # Dictionary 2 (Compound): Multi-phoneme combinations (2+ sounds)
        
        # 1. All standard Sanskrit conjuncts (samyuktaksharas)
        all_conjuncts = [
            # Very common conjuncts
            'क्ष', 'त्र', 'ज्ञ', 'श्र',
            
            # Consonant + य combinations
            'क्य', 'ख्य', 'ग्य', 'घ्य', 'च्य', 'ज्य', 'ञ्य', 'ट्य', 'ड्य', 'ण्य',
            'त्य', 'थ्य', 'द्य', 'ध्य', 'न्य', 'प्य', 'ब्य', 'भ्य', 'म्य', 'श्य', 'ष्य', 'स्य', 'ह्य',
            
            # Consonant + र combinations
            'क्र', 'ख्र', 'ग्र', 'घ्र', 'च्र', 'ज्र', 'ट्र', 'ड्र', 'ण्र',
            'थ्र', 'द्र', 'ध्र', 'प्र', 'ब्र', 'भ्र', 'म्र', 'व्र', 'ष्र', 'स्र', 'ह्र',
            
            # Consonant + व combinations
            'क्व', 'ख्व', 'ग्व', 'घ्व', 'च्व', 'ज्व', 'ट्व', 'ड्व', 'ण्व',
            'त्व', 'थ्व', 'द्व', 'ध्व', 'प्व', 'भ्व', 'म्व', 'श्व', 'ष्व', 'स्व', 'ह्व',
            
            # Consonant + ल combinations
            'क्ल', 'ग्ल', 'च्ल', 'प्ल', 'ब्ल', 'भ्ल', 'म्ल', 'व्ल', 'श्ल', 'स्ल',
            
            # Double consonants (gemination)
            'क्क', 'ग्ग', 'च्च', 'ज्ज', 'ट्ट', 'ड्ड', 'त्त', 'द्द', 'न्न', 'प्प', 'ब्ब', 'म्म',
            'य्य', 'र्र', 'ल्ल', 'व्व', 'श्श', 'ष्ष', 'स्स', 'ह्ह',
            
            # Nasal + consonant combinations
            'ङ्क', 'ङ्ख', 'ङ्ग', 'ङ्घ',
            'ञ्च', 'ञ्छ', 'ञ्ज', 'ञ्झ',
            'ण्ट', 'ण्ठ', 'ण्ड', 'ण्ढ',
            'न्त', 'न्थ', 'न्द', 'न्ध',
            'म्प', 'म्फ', 'म्ब', 'म्भ',
            
            # Consonant + न combinations
            'क्न', 'ग्न', 'च्न', 'ज्न', 'ट्न', 'ड्न', 'त्न', 'द्न', 'प्न', 'ब्न', 'म्न', 'व्न', 'श्न', 'ष्न', 'स्न',
            
            # Consonant + म combinations
            'ग्म', 'द्म', 'त्म', 'प्म', 'ब्म', 'भ्म', 'व्म', 'श्म', 'ष्म', 'स्म',
            
            # Other common 2-consonant combinations
            'स्क', 'स्ख', 'स्त', 'स्थ', 'स्प', 'स्फ',
            'ह्ल', 'ह्म', 'ह्न',
            'ल्य', 'ल्प',
        ]
        vocab.extend(all_conjuncts)
        
        # 2. Triple consonant conjuncts (3-phoneme combinations)
        triple_conjuncts = [
            'क्ष्य', 'क्ष्म', 'क्ष्व',
            'त्र्य', 'त्स्य', 'त्स्न',
            'द्र्य', 'द्व्य',
            'न्त्य', 'न्त्र', 'न्द्र', 'न्द्य',
            'प्र्य', 'प्स्य',
            'स्त्र', 'स्त्य', 'स्थ्य', 'स्प्र', 'स्क्र',
            'श्र्य', 'श्च्य', 'श्व्य',
            'ष्ट्र', 'ष्ट्य', 'ष्प्र', 'ष्ण्य',
            'स्व्य', 'स्म्य',
        ]
        vocab.extend(triple_conjuncts)
        
        # 3. Quadruple consonant conjuncts (4-phoneme combinations)
        quadruple_conjuncts = [
            'न्त्र्य', 'स्त्र्य',
        ]
        vocab.extend(quadruple_conjuncts)
        
        # 4. Conjuncts with vowel matras (consonant cluster + vowel)
        conjunct_with_vowels = []
        common_bases = ['क्ष', 'त्र', 'ज्ञ', 'श्र', 'स्त', 'प्र', 'द्र', 'ग्र', 'ब्र']
        for base in common_bases:
            for matra in self.VOWEL_SIGNS.keys():
                conjunct_with_vowels.append(base + matra)
        vocab.extend(conjunct_with_vowels)
        
        # 5. Vowel-consonant sequences (for Sandhi results)
        vowel_consonant_combos = []
        common_vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ओ']
        common_consonants = ['क', 'ग', 'च', 'ज', 'ट', 'ड', 'त', 'द', 'न', 'प', 'ब', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह']
        for vowel in common_vowels:
            for consonant in common_consonants:
                # Vowel + consonant
                vowel_consonant_combos.append(vowel + consonant)
                # Vowel + consonant with halant
                vowel_consonant_combos.append(vowel + consonant + self.HALANT)
        vocab.extend(vowel_consonant_combos)
        
        # 6. Common Sandhi results (vowel combinations from Sandhi rules)
        sandhi_results = [
            # From vowel + vowel sandhi
            'आ' + 'इ', 'आ' + 'उ', 'आ' + 'ए', 'आ' + 'ओ',
            # Visarga + consonant results
            'अः' + 'क', 'अः' + 'त', 'अः' + 'प', 'अः' + 'च',
        ]
        vocab.extend(sandhi_results)
        
        # 7. Special 2-phoneme patterns with anusvara/visarga
        special_combos = []
        for consonant in common_consonants:
            special_combos.append(consonant + self.ANUSVARA)
            special_combos.append(consonant + self.VISARGA)
            special_combos.append(consonant + 'ं' + 'त')  # Common anusvara patterns
            special_combos.append(consonant + 'ं' + 'स')
        vocab.extend(special_combos)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(vocab))
    
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
            self.token_to_id.get(phoneme, self.token_to_id[self.UNK_TOKEN])
            for phoneme in phonemes
        ]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.token_to_id[self.PAD_TOKEN]] * pad_length)
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
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments passed to encode()
            
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.encode(text, **kwargs)
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
        }
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_phoneme_count(self, text: str) -> int:
        """Get the number of phonemes in text."""
        text = self.normalize_text(text)
        words = text.split()
        count = 0
        for word in words:
            count += len(self.decompose_word(word))
        return count


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SanskritTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"First 20 tokens: {tokenizer.vocab[:20]}")
    
    # Test encoding
    test_text = "राम"
    print(f"\nTest text: {test_text}")
    
    phonemes = tokenizer.decompose_word(test_text)
    print(f"Phonemes: {phonemes}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded IDs: {encoded['input_ids'][:10]}")
    print(f"Attention mask: {encoded['attention_mask'][:10]}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")
    
    # Test longer text
    test_sentence = "नमस्ते भारत"
    print(f"\nTest sentence: {test_sentence}")
    encoded_sentence = tokenizer.encode(test_sentence, max_length=50)
    print(f"Phoneme count: {tokenizer.get_phoneme_count(test_sentence)}")
    print(f"Encoded length: {len([x for x in encoded_sentence['attention_mask'] if x == 1])}")
