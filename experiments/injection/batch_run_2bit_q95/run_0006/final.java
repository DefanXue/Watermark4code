public boolean checkRepeating(String str) {
      // Handle edge case
    if (str =   =    null || str.isEmpty())
{
        return false;  
 }
    return str.matches("\\d+|\\d+-\\d+");
// Check boundary conditions
}