protected NumericType evaluate(NumericType firstNumber, NumericType secondNumber)
    {
        // If either of the arguments is a real number, then use real number arithmetic, otherwise use integer arithmetic.
        if (firstNumber.isInteger() && secondNumber.isInteger())
        {
            long n1 = firstNumber.longValue(); // Use long for intermediate calculations to avoid overflow
            long n2 = secondNumber.longValue();

            long result;
            if (n2 > 0)
            {
                long base = n1;
                long exp = n2;
                long value = 1;
                
                // Using exponentiation by squaring (binary exponentiation) for efficiency
                while (exp > 0) {
                    if (exp % 2 == 1) {
                        value *= base;
                    }
                    base *= base;
                    exp /= 2;
                }
                result = value;
            }
            else
            {
                // Preserve original behavior for non-positive exponents (result remains 1)
                result = 1;
            }

            // If the result fits within an int, return IntLiteral, otherwise return a DoubleLiteral
            if (result >= Integer.MIN_VALUE && result <= Integer.MAX_VALUE) {
                return new IntLiteral((int) result);
            } else {
                return new DoubleLiteral((double) result);
            }
        }
        else
        {
            return new DoubleLiteral(Math.pow(firstNumber.doubleValue(), secondNumber.doubleValue()));
        }
    }