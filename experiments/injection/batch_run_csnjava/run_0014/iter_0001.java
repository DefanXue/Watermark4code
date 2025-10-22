protected NumericType evaluate(NumericType firstNumber, NumericType secondNumber)
    {
        // If either of the arguments is a real number, then use real number arithmetic, otherwise use integer arithmetic.
        if (firstNumber.isInteger() && secondNumber.isInteger())
        {
            int n1 = firstNumber.intValue();
            int n2 = secondNumber.intValue();

            int result = 1;

            if (n2 > 0)
            {
                int base = n1;
                int exp = n2;

                while (exp > 0)
                {
                    if ((exp & 1) == 1)
                    {
                        result *= base;
                    }
                    exp >>= 1;
                    if (exp > 0)
                    {
                        base *= base;
                    }
                }
            }

            return new IntLiteral(result);
        }
        else
        {
            return new DoubleLiteral(Math.pow(firstNumber.doubleValue(), secondNumber.doubleValue()));
        }
    }