protected NumericType evaluate(NumericType first, NumericType second)
{
    // Integer exponentiation when both operands are integers; otherwise real exponentiation.
    if (first.isInteger() && second.isInteger())
    {
        int base = first.intValue();
        int exp = second.intValue();
        int result;

        if (exp > 0)
        {
            // Use BigInteger to compute base^exp and then cast to int to preserve overflow semantics
            java.math.BigInteger bi = java.math.BigInteger.valueOf(base);
            java.math.BigInteger pow = bi.pow(exp);
            result = pow.intValue();
        }
        else
        {
            result = 1;
        }

        return new IntLiteral(result);
    }
    else
    {
        return new DoubleLiteral(Math.pow(first.doubleValue(), second.doubleValue()));
    }
}