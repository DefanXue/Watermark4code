"""Java variable renamer for testing watermark robustness against rename attacks."""

import re
import random
import string
from typing import List, Dict, Set
from .attack_config import AttackConfig


class JavaVariableRenamer:
    """Renames variables in Java code while preserving semantics."""
    
    # Java keywords that should not be renamed
    JAVA_KEYWORDS = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
        'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
        'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
        'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package',
        'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
        'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
        'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
    }
    
    # Common method names that should not be renamed
    COMMON_METHODS = {
        'main', 'toString', 'equals', 'hashCode', 'clone', 'finalize',
        'notify', 'notifyAll', 'wait', 'getClass'
    }
    
    def __init__(self, code: str):
        """Initialize renamer with Java code.
        
        Args:
            code: Java source code to process
        """
        self.code = code
        self.variables = self._extract_variables()
    
    def _extract_variables(self) -> Set[str]:
        """Extract variable names from Java code.
        
        Returns:
            Set of variable names found in the code
        """
        variables = set()
        
        # Pattern to match variable declarations
        # Matches: type varName = ...
        declaration_pattern = r'\b(?:int|long|short|byte|float|double|boolean|char|String|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Pattern to match for-loop variables
        # Matches: for (type var : ...)
        for_pattern = r'\bfor\s*\(\s*(?:int|long|short|byte|float|double|boolean|char|String|var)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]'
        
        # Find all matches
        for match in re.finditer(declaration_pattern, self.code):
            var_name = match.group(1)
            if var_name not in self.JAVA_KEYWORDS and var_name not in self.COMMON_METHODS:
                variables.add(var_name)
        
        for match in re.finditer(for_pattern, self.code):
            var_name = match.group(1)
            if var_name not in self.JAVA_KEYWORDS and var_name not in self.COMMON_METHODS:
                variables.add(var_name)
        
        return variables
    
    def _generate_new_name(self, old_name: str, strategy: str, index: int, seed: int) -> str:
        """Generate a new variable name based on strategy.
        
        Args:
            old_name: Original variable name
            strategy: Naming strategy ('random', 'sequential', 'obfuscated')
            index: Index for sequential naming
            seed: Random seed
            
        Returns:
            New variable name
        """
        if strategy == 'random':
            random.seed(seed + hash(old_name))
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            return f'var_{suffix}'
        
        elif strategy == 'sequential':
            return f'v{index}'
        
        elif strategy == 'obfuscated':
            # Use confusing characters: l (lowercase L), O (uppercase O), I (uppercase i)
            confusing_chars = ['l', 'O', 'I', 'll', 'lO', 'OI', 'Il']
            random.seed(seed + hash(old_name))
            return random.choice(confusing_chars) + str(index)
        
        else:
            return old_name
    
    def apply_renames(self, config: AttackConfig) -> str:
        """Apply variable renaming to the code.
        
        Args:
            config: Attack configuration
            
        Returns:
            Renamed Java code
        """
        if not self.variables:
            return self.code
        
        # Create rename mapping
        rename_map: Dict[str, str] = {}
        for idx, var in enumerate(sorted(self.variables)):
            new_name = self._generate_new_name(
                var, 
                config.naming_strategy, 
                idx, 
                config.seed
            )
            rename_map[var] = new_name
        
        # Apply renames
        renamed_code = self.code
        
        # Sort by length (descending) to avoid partial replacements
        # e.g., replace 'count' before 'c' to avoid 'c' -> 'v0' affecting 'count'
        for old_name in sorted(rename_map.keys(), key=len, reverse=True):
            new_name = rename_map[old_name]
            
            # Use word boundary to avoid partial matches
            # This ensures we only replace complete variable names
            pattern = r'\b' + re.escape(old_name) + r'\b'
            renamed_code = re.sub(pattern, new_name, renamed_code)
        
        return renamed_code


def rename_variables(code: str, strategy: str = 'random', seed: int = 42) -> str:
    """Convenience function to rename variables in Java code.
    
    Args:
        code: Java source code
        strategy: Naming strategy ('random', 'sequential', 'obfuscated')
        seed: Random seed for reproducibility
        
    Returns:
        Renamed Java code
    """
    config = AttackConfig(naming_strategy=strategy, seed=seed)
    renamer = JavaVariableRenamer(code)
    return renamer.apply_renames(config)

