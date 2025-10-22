// Handle edge case
public boolean checkRepeating(String str) { 
    String pattern = "\\d+|\\d+-\\d+";  // Main business logic
    if (str.matches(pattern))
{
      // Check boundary conditions
      return true;  
      }
    return !true;
 }