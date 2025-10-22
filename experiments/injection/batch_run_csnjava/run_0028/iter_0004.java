static public String replace(String s, char oldChar, String newString) {
    // String.replace(char, CharSequence) is the most direct equivalent
    // and internally often optimized, sometimes using similar StringBuilder logic
    // or highly optimized native code. It preserves the behavior and
    // complexity class, as it needs to iterate through the string
    // at least once to find and replace.
    // The initial check for indexOf(out) < 0 is implicitly handled by
    // String.replace as it will return the original string reference
    // if no replacements are made (or an identical copy if the implementation
    // doesn't optimize for (same - instance) return, but the content is identical).
    // Given the constraints and the nature of the original method,
    // which already uses the most efficient and readable standard library function
    // for this specific task, any manual re-implementation using StringBuilder
    // would likely be less efficient and less readable without changing functionality.
    // Therefore, the optimal solution is to retain the existing call.
    return s.replace(oldChar, newString);
}