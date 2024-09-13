# tokziner.py

from __future__ import annotations
import ctypes
import functools
import subprocess
import io
import logging
import multiprocessing
from multiprocessing import Pool
from typing import Tuple, TypeVar, List, Any, cast, Set, Optional, Dict
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from numba import jit
import numpy as np
from cryptography.fernet import Fernet

# ============================
# ====== Lore of Tokziner ======
# ============================

# ============================
# ====== Configuration ========
# ============================

# Constants for special tokens
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Tokziner")

# ============================
# ====== Security Module ======
# ============================

class Security:
    """
    Handles encryption and decryption to ensure data privacy and security.
    """
    def __init__(self, key: Optional[bytes] = None):
        """
        Initializes the Security module with a provided key or generates one.
        """
        self.key = key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        logger.debug("Security module initialized.")

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypts the provided data.
        """
        encrypted = self.cipher_suite.encrypt(data)
        logger.debug("Data encrypted.")
        return encrypted

    def decrypt_data(self, data: bytes) -> bytes:
        """
        Decrypts the provided data.
        """
        decrypted = self.cipher_suite.decrypt(data)
        logger.debug("Data decrypted.")
        return decrypted

# ============================
# ====== Tokenizer Class ======
# ============================

T = TypeVar("T")

class Tokziner:
    """
    An advanced tokenizer with rich lore-inspired features.

    Lore: Born from the ancient scripts of Linguaria, Tokziner deciphers the essence of language, preserving its true meaning.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        special_tokens: Optional[List[str]] = None,
        tokenizer_path: str = "tokziner.json",
        model_type: str = "BPE",
        context_size: int = 128,
        encryption_key: Optional[bytes] = None
    ):
        """
        Initializes the Tokziner tokenizer with specified configurations.

        Args:
            vocab_size (int): The size of the vocabulary.
            special_tokens (List[str], optional): List of special tokens.
            tokenizer_path (str): Path to save/load the tokenizer.
            model_type (str): Tokenization model type ('BPE' or 'Unigram').
            context_size (int): The context window size for context-aware tokenization.
            encryption_key (bytes, optional): Key for encrypting sensitive data.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type.upper()
        self.context_size = context_size
        self.security = Security(encryption_key)

        logger.info("Initializing Tokziner...")
        self.tokenizer = self.create_tokenizer()
        self.trainer = self.create_trainer()
        logger.info("Tokziner initialized successfully.")

    def create_tokenizer(self) -> Tokenizer:
        """
        Creates and configures the tokenizer based on the specified model type.

        Returns:
            Tokenizer: Configured tokenizer instance.
        """
        logger.debug(f"Creating tokenizer with model type: {self.model_type}")
        if self.model_type == "BPE":
            model = models.BPE()
        elif self.model_type == "UNIGRAM":
            model = models.Unigram()
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError("Unsupported model type. Choose 'BPE' or 'Unigram'.")

        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder()
        tokenizer.post_processor = processors.BertProcessing(
            ("<EOS>", self.special_tokens.index("<EOS>")),
            ("<BOS>", self.special_tokens.index("<BOS>"))
        )
        logger.debug("Tokenizer configuration completed.")
        return tokenizer

    def create_trainer(self) -> trainers.BpeTrainer:
        """
        Creates a trainer for the tokenizer.

        Returns:
            trainers.BpeTrainer: Configured trainer instance.
        """
        logger.debug("Creating trainer for the tokenizer.")
        if self.model_type == "BPE":
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens,
                show_progress=True,
                initial_alphabet=pre_tokenizers.BertPreTokenizer.alphabet()
            )
        elif self.model_type == "UNIGRAM":
            trainer = trainers.UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens,
                show_progress=True,
                initial_alphabet=pre_tokenizers.BertPreTokenizer.alphabet()
            )
        logger.debug("Trainer configuration completed.")
        return trainer

    def train(self, files: List[str]) -> None:
        """
        Trains the tokenizer on the provided text files.

        Args:
            files (List[str]): List of file paths containing training data.
        """
        logger.info("Starting tokenizer training...")
        self.tokenizer.train(files, self.trainer)
        logger.info("Tokenizer training completed.")

    def save(self) -> None:
        """
        Saves the trained tokenizer to a JSON file.
        """
        logger.info(f"Saving tokenizer to {self.tokenizer_path}...")
        # Encrypt tokenizer data before saving
        raw_data = self.tokenizer.to_str().encode('utf-8')
        encrypted_data = self.security.encrypt_data(raw_data)
        with open(self.tokenizer_path, 'wb') as f:
            f.write(encrypted_data)
        logger.info("Tokenizer saved successfully.")

    @staticmethod
    def load(tokenizer_path: str = "tokziner.json", encryption_key: Optional[bytes] = None) -> Tokziner:
        """
        Loads a tokenizer from a JSON file.

        Args:
            tokenizer_path (str): Path to the tokenizer JSON file.
            encryption_key (bytes, optional): Key for decrypting sensitive data.

        Returns:
            Tokziner: An instance of Tokziner with the loaded tokenizer.
        """
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        security = Security(encryption_key)
        with open(tokenizer_path, 'rb') as f:
            encrypted_data = f.read()
        raw_data = security.decrypt_data(encrypted_data)
        tokenizer = Tokenizer.from_str(raw_data.decode('utf-8'))
        logger.info("Tokenizer loaded successfully.")
        
        # Initialize Tokziner instance with loaded tokenizer
        tokziner = Tokziner()
        tokziner.tokenizer = tokenizer
        return tokziner

    @staticmethod
    def convert_to_transformers(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
        """
        Converts Tokziner to a Hugging Face PreTrainedTokenizerFast for integration with Transformers.

        Args:
            tokenizer (Tokenizer): Tokziner tokenizer instance.

        Returns:
            PreTrainedTokenizerFast: Transformer-compatible tokenizer.
        """
        logger.debug("Converting Tokziner to PreTrainedTokenizerFast.")
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                       unk_token="<UNK>", 
                                       pad_token="<PAD>", 
                                       bos_token="<BOS>", 
                                       eos_token="<EOS>")

    def tokenize_batch(self, texts: List[str], padding: bool = True, truncation: bool = True) -> Dict[str, Any]:
        """
        Tokenizes a batch of texts.

        Args:
            texts (List[str]): List of text strings to tokenize.
            padding (bool): Whether to pad the sequences.
            truncation (bool): Whether to truncate the sequences.

        Returns:
            Dict[str, Any]: Tokenized outputs.
        """
        logger.info("Tokenizing batch of texts.")
        transformers_tokenizer = self.convert_to_transformers(self.tokenizer)
        return transformers_tokenizer.batch_encode_plus(texts, padding=padding, truncation=truncation)

    def tokenize_parallel(self, texts: List[str], processes: int = 4) -> List[Any]:
        """
        Tokenizes texts in parallel using multiprocessing.

        Args:
            texts (List[str]): List of text strings to tokenize.
            processes (int): Number of parallel processes.

        Returns:
            List[Any]: List of tokenized outputs.
        """
        logger.info("Tokenizing texts in parallel.")
        with Pool(processes=processes) as pool:
            return pool.map(self.tokenizer.encode, texts)

    def __call__(self, text: str) -> Dict[str, Any]:
        """
        Tokenizes a single text input.

        Args:
            text (str): Text string to tokenize.

        Returns:
            Dict[str, Any]: Tokenized output.
        """
        logger.info("Tokenizing a single text input.")
        transformers_tokenizer = self.convert_to_transformers(self.tokenizer)
        return transformers_tokenizer.encode_plus(text, add_special_tokens=True)

    def extend_vocabulary(self, new_tokens: List[str]) -> None:
        """
        Extends the tokenizer's vocabulary with new tokens.

        Args:
            new_tokens (List[str]): List of new tokens to add.
        """
        logger.info("Extending vocabulary with new tokens.")
        self.tokenizer.add_tokens(new_tokens)
        logger.info("Vocabulary extended successfully.")

    def secure_tokenize(self, text: str) -> bytes:
        """
        Tokenizes text and encrypts the tokenized data.

        Args:
            text (str): Text string to tokenize.

        Returns:
            bytes: Encrypted tokenized data.
        """
        logger.info("Tokenizing and encrypting text.")
        tokens = self.tokenizer.encode(text).ids
        tokens_bytes = np.array(tokens, dtype=np.int32).tobytes()
        encrypted_tokens = self.security.encrypt_data(tokens_bytes)
        logger.info("Text tokenized and encrypted successfully.")
        return encrypted_tokens

    def secure_detokenize(self, encrypted_tokens: bytes) -> str:
        """
        Decrypts and detokenizes the tokenized data.

        Args:
            encrypted_tokens (bytes): Encrypted tokenized data.

        Returns:
            str: Original text string.
        """
        logger.info("Decrypting and detokenizing tokens.")
        tokens_bytes = self.security.decrypt_data(encrypted_tokens)
        tokens = np.frombuffer(tokens_bytes, dtype=np.int32).tolist()
        text = self.tokenizer.decode(tokens)
        logger.info("Tokens decrypted and text detokenized successfully.")
        return text

# ============================
# ====== Usage Example ========
# ============================

if __name__ == "__main__":
    # Initialize Tokziner with default settings
    tokziner = Tokziner()

    # Path to your training data
    training_files = ["data/text_corpus.txt"]  # Ensure this file exists with your corpus

    # Train the tokenizer
    tokziner.train(training_files)

    # Save the trained tokenizer securely
    tokziner.save()

    # Load the tokenizer (for demonstration)
    loaded_tokziner = Tokziner.load()

    # Tokenize a single text input
    sample_text = "Welcome to the mystical world of Tokziner!"
    encoding = loaded_tokziner(sample_text)
    logger.info(f"Tokens: {encoding['input_ids']}")
    logger.info(f"Decoded Text: {loaded_tokziner.tokenizer.decode(encoding['input_ids'])}")

    # Tokenize a batch of texts
    batch_texts = [
        "First sentence in the realm.",
        "Second sentence echoes through Linguaria.",
        "Third sentence preserves the legacy."
    ]
    batch_encodings = loaded_tokziner.tokenize_batch(batch_texts)
    logger.info(f"Batch Encodings: {batch_encodings}")

    # Tokenize texts in parallel
    parallel_encodings = loaded_tokziner.tokenize_parallel(batch_texts)
    logger.info(f"Parallel Encodings: {parallel_encodings}")

    # Secure tokenization and detokenization
    encrypted = loaded_tokziner.secure_tokenize(sample_text)
    logger.info(f"Encrypted Tokens: {encrypted}")

    decrypted_text = loaded_tokziner.secure_detokenize(encrypted)
    logger.info(f"Decrypted Text: {decrypted_text}")

    # Extend vocabulary with new tokens
    new_tokens = ["<NEW1>", "<NEW2>", "<NEW3>"]
    loaded_tokziner.extend_vocabulary(new_tokens)
    logger.info(f"Extended Vocabulary: {loaded_tokziner.tokenizer.get_vocab()}")

    # Tokenize after extending vocabulary
    extended_encoding = loaded_tokziner("<NEW1> Welcome!")
    logger.info(f"Extended Encoding: {extended_encoding['input_ids']}")
    logger.info(f"Decoded Extended Text: {loaded_tokziner.tokenizer.decode(extended_encoding['input_ids'])}")
