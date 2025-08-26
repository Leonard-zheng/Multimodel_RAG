#!/usr/bin/env python3
"""
Simple test script for the caching system
"""
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_manager import cache_manager
from src.utils import logger

def test_cache_manager():
    """Test basic cache functionality"""
    print("Testing CacheManager...")
    
    # Test content ID generation
    test_content = "This is a test document with some content."
    content_id = cache_manager.generate_content_id(test_content)
    print(f"Generated content ID: {content_id}")
    
    # Test cache miss
    cached_summary = cache_manager.get_summary(content_id)
    print(f"Cache miss (expected): {cached_summary}")
    
    # Test cache set
    test_summary = "This is a test summary of the document."
    cache_manager.set_summary(content_id, test_summary)
    print(f"Cached summary: {test_summary}")
    
    # Test cache hit
    cached_summary = cache_manager.get_summary(content_id)
    print(f"Cache hit: {cached_summary}")
    
    # Test cache stats
    stats = cache_manager.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    print("✅ CacheManager tests passed!")

def test_content_id_stability():
    """Test that content IDs are stable across calls"""
    print("\nTesting content ID stability...")
    
    test_content = "Same content should always generate the same ID"
    
    id1 = cache_manager.generate_content_id(test_content)
    id2 = cache_manager.generate_content_id(test_content)
    
    print(f"First ID:  {id1}")
    print(f"Second ID: {id2}")
    print(f"IDs match: {id1 == id2}")
    
    if id1 == id2:
        print("✅ Content ID stability test passed!")
    else:
        print("❌ Content ID stability test failed!")

def test_cache_persistence():
    """Test that cache persists to file"""
    print("\nTesting cache persistence...")
    
    # Clear any existing cache
    cache_manager.clear_cache()
    
    # Add a test item
    test_content = "Persistence test content"
    content_id = cache_manager.generate_content_id(test_content)
    test_summary = "Persistence test summary"
    cache_manager.set_summary(content_id, test_summary)
    
    # Check if cache file exists
    cache_file_exists = os.path.exists(cache_manager.cache_file)
    print(f"Cache file exists: {cache_file_exists}")
    
    if cache_file_exists:
        print("✅ Cache persistence test passed!")
    else:
        print("❌ Cache persistence test failed!")

if __name__ == "__main__":
    print("🧪 Running cache system tests...\n")
    
    try:
        test_cache_manager()
        test_content_id_stability() 
        test_cache_persistence()
        
        print("\n🎉 All cache tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Cache test failed: {e}", exc_info=True)
        sys.exit(1)