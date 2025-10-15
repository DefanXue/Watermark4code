public boolean checkRepeating(String str) {
  if (str == null || str.isEmpty()) {
    return false;
  } else {
    for (char c : str.toCharArray()) {
      String tmp = "" + c + "<event>";
    }
    return str.matches("\\d+|\\d+-\\d+");
  }
}