protected NumericType evaluate(NumericType first, NumericType second)
{
    // Integer exponentiation when both operands are integers; otherwise real exponentiation.
    if (first.isInteger() && second.isInteger())
    {
        int base = first.intValue();
        int exp = second.intValue();
        int result = 1;

        if (exp > 0)
        {
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
        return new DoubleLiteral(Math.pow(first.doubleValue(), second.doubleValue()));
    }
}