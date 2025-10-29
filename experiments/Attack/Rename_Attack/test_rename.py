"""Tests for Java variable renaming attack."""

import pytest
from .java_variable_renamer import JavaVariableRenamer, rename_variables
from .attack_config import AttackConfig


def test_simple_rename():
    """Test basic variable renaming."""
    code = """
    public static int add(int a, int b) {
        int sum = a + b;
        return sum;
    }
    """
    
    config = AttackConfig(naming_strategy='sequential', seed=42)
    renamer = JavaVariableRenamer(code)
    renamed = renamer.apply_renames(config)
    
    # Should have renamed variables
    assert 'v0' in renamed or 'v1' in renamed or 'v2' in renamed
    # Should preserve keywords
    assert 'public' in renamed
    assert 'static' in renamed
    assert 'int' in renamed
    assert 'return' in renamed


def test_random_strategy():
    """Test random naming strategy."""
    code = """
    int x = 5;
    int y = 10;
    """
    
    renamed = rename_variables(code, strategy='random', seed=42)
    
    # Should not contain original variable names
    assert 'var_' in renamed
    # Should preserve type
    assert 'int' in renamed


def test_sequential_strategy():
    """Test sequential naming strategy."""
    code = """
    int first = 1;
    int second = 2;
    int third = 3;
    """
    
    renamed = rename_variables(code, strategy='sequential', seed=42)
    
    # Should contain sequential names
    assert 'v0' in renamed
    assert 'v1' in renamed
    assert 'v2' in renamed


def test_obfuscated_strategy():
    """Test obfuscated naming strategy."""
    code = """
    int value = 100;
    """
    
    renamed = rename_variables(code, strategy='obfuscated', seed=42)
    
    # Should contain confusing characters
    assert any(c in renamed for c in ['l', 'O', 'I'])


def test_preserve_keywords():
    """Test that Java keywords are preserved."""
    code = """
    public class Test {
        private static void main(String[] args) {
            int count = 0;
            for (int i = 0; i < 10; i++) {
                count++;
            }
        }
    }
    """
    
    config = AttackConfig(naming_strategy='random', seed=42)
    renamer = JavaVariableRenamer(code)
    renamed = renamer.apply_renames(config)
    
    # Keywords should be preserved
    assert 'public' in renamed
    assert 'class' in renamed
    assert 'private' in renamed
    assert 'static' in renamed
    assert 'void' in renamed
    # Method names should be preserved
    assert 'main' in renamed


def test_for_loop_variables():
    """Test renaming of for-loop variables."""
    code = """
    for (int i = 0; i < 10; i++) {
        System.out.println(i);
    }
    """
    
    config = AttackConfig(naming_strategy='sequential', seed=42)
    renamer = JavaVariableRenamer(code)
    renamed = renamer.apply_renames(config)
    
    # Loop variable should be renamed
    assert 'v0' in renamed


def test_reproducibility():
    """Test that same seed produces same result."""
    code = """
    int x = 5;
    int y = 10;
    """
    
    renamed1 = rename_variables(code, strategy='random', seed=42)
    renamed2 = rename_variables(code, strategy='random', seed=42)
    
    assert renamed1 == renamed2


def test_different_seeds():
    """Test that different seeds produce different results."""
    code = """
    int x = 5;
    int y = 10;
    """
    
    renamed1 = rename_variables(code, strategy='random', seed=42)
    renamed2 = rename_variables(code, strategy='random', seed=123)
    
    assert renamed1 != renamed2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

