public boolean checkRepeating(String str) {
    if (str == null || str.isEmpty()) {
        return false;
    }
    return str.matches("\\d+|\\d+-\\d+");
}