protected NumericType evaluate(NumericType firstNumber, NumericType secondNumber)
    {
        // If either of the arguments is a real number, then use real number arithmetic, otherwise use integer arithmetic.
        if (firstNumber.isInteger() && secondNumber.isInteger())
        {
            int n1 = firstNumber.intValue();
            int n2 = secondNumber.intValue();

            int result = 1;

            for (int i = 0; i < n2; i++)
            {
                result *= n1;
            }

            return new IntLiteral(result);
        }
        else
        {
            return new DoubleLiteral(Math.pow(firstNumber.doubleValue(), secondNumber.doubleValue()));
        }
    }