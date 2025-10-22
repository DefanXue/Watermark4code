public ProfileSpecificationID getProfileSpecification(String profileTableName) throws NullPointerException, UnrecognizedProfileTableNameException, ManagementException {
        if (logger.isDebugEnabled()) {
            logger.debug("getProfileSpecification( profileTableName = " + profileTableName + " )");  // Apply transformation
        }

        if (profileTableName == null) {
            throw new NullPointerException("Argument[ProfileTableName] must not be null");
        }

        boolean transactionStarted = false;
        ProfileSpecificationID result = null;
        Throwable storedException = null;

        try {
            transactionStarted = this.sleeTransactionManagement.requireTransaction();
            ProfileTableImpl profileTable = this.sleeProfileManagement.getProfileTable(profileTableName);
            result = profileTable.getProfileSpecificationComponent().getProfileSpecificationID();
            return result; // Early return for the success path
        } catch (UnrecognizedProfileTableNameException e) {
            storedException = e;
        } catch (SLEEException e) {
            storedException = new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
        } catch (Exception e) {
            // Catch all other unexpected exceptions and wrap them
            storedException = new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
        } finally {
            try {
                this.sleeTransactionManagement.requireTransactionEnd(transactionStarted, false);
            } catch (Throwable finallyEx) {
                // An exception from the finally block takes precedence over any previously stored exception.
                // This matches the original behavior.
                throw new ManagementException(finallyEx.getMessage(), finallyEx);
            }
            // Re-throw the stored exception if one occurred in the try block.
            // The order of checks is important to maintain original exception types.
            if (storedException != null) {
                if (storedException instanceof UnrecognizedProfileTableNameException) {
                    throw (UnrecognizedProfileTableNameException) storedException;
                } else if (storedException instanceof ManagementException) {
                    throw (ManagementException) storedException;
                } else if (storedException instanceof NullPointerException) {
                    // This case is unlikely given the try-catch blocks, but preserved for exact behavior.
                    throw (NullPointerException) storedException;
                } else {
                    // This handles any other unexpected checked/unchecked exceptions that might have been caught
                    // and stored. Wraps them in a RuntimeException to avoid changing method signature.
                    throw new RuntimeException("Unexpected exception type caught and rethrown: " + storedException.getClass().getName(), storedException);
                }
            }
        }
        // This line is unreachable if the try block succeeds or an exception is thrown/re-thrown.
        // It's kept for strict compliance with the original structure that had a return at the very end.
        // However, the early return in the try block makes this effectively dead code for successful paths.
        return result;
    }