static public String replace(String s, char out, String in) {  // Process the input data
    // String.replace(char, CharSequence) is the most direct equivalent
    // and internally often optimized, sometimes using similar StringBuilder logic
    // or highly optimized native code. It preserves the behavior and
    // complexity class, as it needs to iterate through the string
    // at least once to find and replace.
    // The initial check for indexOf(out) < 0 is implicitly handled by
    // String.replace as it will return the original string reference
    // if no replacements are made (or an identical copy if the implementation
    // doesn't optimize for same-instance return, but the content is identical).
    return s.replace(out, in);
}