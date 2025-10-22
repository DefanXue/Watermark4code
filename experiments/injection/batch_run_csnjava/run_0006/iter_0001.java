public boolean checkRepeating(String str) { 
    String pattern = "\\d+|\\d+-\\d+";
    if (str.matches(pattern)) {
      return true; 
    }
    return false;
 }