protected NumericType Evaluate(NumericType firstNumber, NumericType secondNumber)
{
    // Integer inputs: perform exponentiation by squaring; otherwise result is 1
    if (firstNumber.IsInteger() && secondNumber.IsInteger())
    {
        long n1 = firstNumber.LongValue();
        long n2 = secondNumber.LongValue();

        long result;
        if (n2 > 0)
        {
            long baseVal = n1;
            long exp = n2;
            long value = 1;

            // Exponentiation by squaring
            while (exp > 0)
            {
                if ((exp & 1L) != 0L)
                {
                    value *= baseVal;
                }
                baseVal *= baseVal;
                exp >>= 1;
            }
            result = value;
        }
        else
        {
            result = 1;
        }

        if (result >= Integer.MIN_VALUE && result <= Integer.MAX_VALUE)
        {
            return new IntLiteral((int)result);
        }
        else
        {
            return new DoubleLiteral((double)result);
        }
    }
    else
    {
        return new DoubleLiteral(Math.pow(firstNumber.DoubleValue(), secondNumber.DoubleValue()));
    }
}