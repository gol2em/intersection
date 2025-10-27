"""
Test to verify CRIT logic bug.

CRIT = 0.8 means:
- If PP reduces box by MORE than 20%, DON'T subdivide (apply PP again)
- If PP reduces box by LESS than 20%, DO subdivide

Example:
- Original box: [0, 1], width = 1.0
- PP bounds: [0.121, 0.879], width = 0.758
- Reduction: (1.0 - 0.758) / 1.0 = 24.2%
- Since 24.2% > 20%, should NOT subdivide
- Should extract sub-box [0.121, 0.879] and apply PP again

Current buggy logic:
- containing_ranges = [0.121, 0.879] (in [0,1] space)
- size = 0.758
- Check: if size > crit → if 0.758 > 0.8 → FALSE
- dims_to_subdivide = [] (empty)
- Falls into "all dimensions small enough" case
- Subdivides anyway!

Correct logic:
- original_size = 1.0 (the box we're working on)
- pp_size = 0.758
- ratio = pp_size / original_size = 0.758
- Check: if ratio > crit → if 0.758 > 0.8 → FALSE → DON'T subdivide
- Extract sub-box and apply PP again
"""

import numpy as np


def test_crit_logic_buggy():
    """Current buggy implementation."""
    print("=" * 80)
    print("BUGGY LOGIC (Current Implementation)")
    print("=" * 80)
    
    crit = 0.8
    containing_ranges = [(0.121212, 0.878788)]  # PP bounds in [0,1] space
    
    dims_to_subdivide = []
    for i, (t_min, t_max) in enumerate(containing_ranges):
        size = t_max - t_min
        print(f"Dimension {i}: size = {size:.6f}")
        print(f"  Check: size > crit → {size:.6f} > {crit:.6f} → {size > crit}")
        
        if size > crit:
            dims_to_subdivide.append(i)
    
    print(f"\nDimensions to subdivide: {dims_to_subdivide}")
    
    if not dims_to_subdivide:
        print("  → All dimensions 'small enough'")
        print("  → Forcing subdivision along largest dimension!")
        sizes = [t_max - t_min for t_min, t_max in containing_ranges]
        dims_to_subdivide = [np.argmax(sizes)]
        print(f"  → dims_to_subdivide = {dims_to_subdivide}")
    
    print(f"\nResult: SUBDIVIDE (WRONG!)")
    print()


def test_crit_logic_correct():
    """Correct implementation."""
    print("=" * 80)
    print("CORRECT LOGIC (Should Be)")
    print("=" * 80)
    
    crit = 0.8
    original_size = 1.0  # Original box [0, 1]
    containing_ranges = [(0.121212, 0.878788)]  # PP bounds in [0,1] space
    
    dims_to_subdivide = []
    for i, (t_min, t_max) in enumerate(containing_ranges):
        pp_size = t_max - t_min
        ratio = pp_size / original_size
        reduction_pct = (1 - ratio) * 100
        
        print(f"Dimension {i}:")
        print(f"  Original size: {original_size:.6f}")
        print(f"  PP size: {pp_size:.6f}")
        print(f"  Ratio: {ratio:.6f}")
        print(f"  Reduction: {reduction_pct:.2f}%")
        print(f"  Check: ratio > crit → {ratio:.6f} > {crit:.6f} → {ratio > crit}")
        
        if ratio > crit:
            # PP didn't reduce enough (< 20% reduction)
            dims_to_subdivide.append(i)
            print(f"  → PP reduction < 20%, SUBDIVIDE")
        else:
            # PP reduced significantly (≥ 20% reduction)
            print(f"  → PP reduction ≥ 20%, DON'T subdivide, apply PP again")
    
    print(f"\nDimensions to subdivide: {dims_to_subdivide}")
    
    if not dims_to_subdivide:
        print(f"\nResult: DON'T SUBDIVIDE, extract sub-box and apply PP again (CORRECT!)")
    else:
        print(f"\nResult: SUBDIVIDE")
    print()


def test_case_should_subdivide():
    """Test case where PP doesn't help much."""
    print("=" * 80)
    print("TEST CASE: PP Reduction < 20% (Should Subdivide)")
    print("=" * 80)
    
    crit = 0.8
    original_size = 1.0
    containing_ranges = [(0.1, 0.95)]  # PP only reduced to 0.85 width
    
    pp_size = 0.95 - 0.1
    ratio = pp_size / original_size
    reduction_pct = (1 - ratio) * 100
    
    print(f"Original size: {original_size:.6f}")
    print(f"PP size: {pp_size:.6f}")
    print(f"Ratio: {ratio:.6f}")
    print(f"Reduction: {reduction_pct:.2f}%")
    print(f"Check: ratio > crit → {ratio:.6f} > {crit:.6f} → {ratio > crit}")
    
    if ratio > crit:
        print(f"\n✓ CORRECT: Should subdivide (PP reduction < 20%)")
    else:
        print(f"\n✗ WRONG: Should subdivide but didn't")
    print()


if __name__ == "__main__":
    test_crit_logic_buggy()
    test_crit_logic_correct()
    test_case_should_subdivide()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("BUG IDENTIFIED:")
    print("  The CRIT check compares PP size against absolute threshold 0.8")
    print("  instead of comparing the RATIO of (PP size / original size)")
    print()
    print("CONSEQUENCE:")
    print("  When PP reduces box from 1.0 to 0.758 (24.2% reduction),")
    print("  the code thinks 0.758 < 0.8 means 'small enough'")
    print("  and then FORCES subdivision anyway!")
    print()
    print("FIX:")
    print("  Compare ratio = (PP size / original size) against CRIT")
    print("  If ratio > CRIT: subdivide (PP didn't help much)")
    print("  If ratio ≤ CRIT: don't subdivide (PP helped, apply again)")
    print()

