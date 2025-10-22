public ProfileSpecificationID getProfileSpecification(String profileTableName) throws NullPointerException, UnrecognizedProfileTableNameException, ManagementException {
        if (logger.isDebugEnabled()) {
            logger.debug("getProfileSpecification( profileTableName = " + profileTableName + " )");
        }

        if (profileTableName == null) {
            throw new NullPointerException("Argument[ProfileTableName] must not be null");
        }

        boolean transactionStarted = false;
        ProfileSpecificationID result = null; // Initialize result to null
        Throwable caughtException = null; // To store any exception from the try block

        try {
            transactionStarted = this.sleeTransactionManagement.requireTransaction();
            ProfileTableImpl profileTable = this.sleeProfileManagement.getProfileTable(profileTableName);
            result = profileTable.getProfileSpecificationComponent().getProfileSpecificationID();
            // Do not return here. Store the result and let the finally block handle transaction end.
        } catch (UnrecognizedProfileTableNameException e) {
            caughtException = e; // Store the exception
        } catch (SLEEException e) {
            caughtException = new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
        } catch (Exception e) {
            caughtException = new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
        } finally {
            try {
                this.sleeTransactionManagement.requireTransactionEnd(transactionStarted, false);
            } catch (Throwable e) {
                // If an exception was already caught in the try block,
                // this new exception from finally block takes precedence.
                // This matches the original behavior where a finally block exception
                // would supersede any pending exception from the try block.
                throw new ManagementException(e.getMessage(), e);
            }

            // If an exception was caught in the try block, re-throw it after the finally block executes.
            if (caughtException != null) {
                if (caughtException instanceof UnrecognizedProfileTableNameException) {
                    throw (UnrecognizedProfileTableNameException) caughtException;
                } else if (caughtException instanceof ManagementException) {
                    throw (ManagementException) caughtException;
                } else if (caughtException instanceof NullPointerException) { // Although unlikely to be caught here, for completeness
                    throw (NullPointerException) caughtException;
                } else {
                    // This case should ideally not be reached if all specific exceptions are handled.
                    // However, if a new unchecked exception type were introduced, this would catch it.
                    // Given the original structure, all caught exceptions are either UPTNE or wrapped into ManagementException.
                    // For safety, re-throw as a generic RuntimeException if it's not one of the expected checked types.
                    // This is a defensive measure to ensure no exception is swallowed.
                    throw new RuntimeException("Unexpected exception type caught and rethrown: " + caughtException.getClass().getName(), caughtException);
                }
            }
        }
        // If no exception was thrown, return the result.
        return result;
    }