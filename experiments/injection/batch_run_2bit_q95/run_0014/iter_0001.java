protected NumericType evaluate(NumericType firstNumber, NumericType secondNumber)
    {
        // If either of the arguments is a real number, then use real number arithmetic, otherwise use integer arithmetic.
        if (firstNumber.isInteger() && secondNumber.isInteger())
        {
            int n1 = firstNumber.intValue();
            int n2 = secondNumber.intValue();

            int result;
            if (n2 > 0)
            {
                int base = n1;
                int exp = n2;
                int value = 1;
                while (exp > 0)
                {
                    if ((exp & 1) == 1)
                    {
                        value *= base;
                    }
                    base *= base;
                    exp >>= 1;
                }
                result = value;
            }
            else
            {
                // Preserve original behavior for non-positive exponents (result remains 1)
                result = 1;
            }

            return new IntLiteral(result);
        }
        else
        {
            return new DoubleLiteral(Math.pow(firstNumber.doubleValue(), secondNumber.doubleValue()));
        }
    }