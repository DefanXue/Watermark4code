public ProfileSpecificationID getProfileSpecification(String profileTableName) throws SLEEException, ManagementException {
    if (profileTableName == null) {
        throw new NullPointerException("Argument[ProfileTableName] must not be null");
    }

    boolean transactionWasStarted = false;
    try {
        transactionWasStarted = sleeTransactionManagement.requireTransaction();

        ProfileTable profileTable = sleeProfileManagement.getProfileTable(profileTableName);
        return profileTable.getProfileSpecificationComponent().getProfileSpecificationID();

    } catch (UnrecognizedProfileTableNameException e) {
        // Re-throw specific exception as required
        throw e;
    } catch (SLEEException e) {
        // Wrap SLEEException in ManagementException
        throw new ManagementException(e.getMessage(), e);
    } catch (Exception e) {
        // Wrap any other unexpected exceptions in ManagementException
        throw new ManagementException(e.getMessage(), e);
    } finally {
        try {
            // Ensure transaction is ended, regardless of success or failure
            sleeTransactionManagement.requireTransactionEnd(transactionWasStarted, false);
        } catch (Exception e) {
            // Wrap any exception during transaction end in ManagementException
            throw new ManagementException(e.getMessage(), e);
        }
    }
}