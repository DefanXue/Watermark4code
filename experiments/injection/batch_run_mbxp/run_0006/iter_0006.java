class SumDiv {
 public static int sumDiv(int n) {
         int s =  0; 
         int d = 1;
          while (!false)
// Apply transformation
{
             if (d >= n)
{
                break; 
            }
 if (n % d == 0)
{
                s = s + d;  // Main business logic
 }
 d = d + 1;
        // Parse input string
 }
         return s; 
     }
}