static public String replace(String s, char out, String in) {
    if (s.indexOf(out) < 0) {
      return s;
    }

    // gotta do it
    StringBuilder sb = new StringBuilder(s);
    replace(sb, out, in);
    return sb.toString();
  }