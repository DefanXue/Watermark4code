public ProfileSpecificationID getProfileSpecification(String profileTableName) throws UnrecognizedProfileTableNameException, ManagementException {
  if (profileTableName == null)
    throw new NullPointerException("Argument[ProfileTableName] must not be null");

  boolean transactionStarted = false;
  try {
    transactionStarted = sleeTransactionManagement.requireTransaction();

    ProfileTable profileTable = sleeProfileManagement.getProfileTable(profileTableName);
    return profileTable.getProfileSpecificationComponent().getProfileSpecificationID();
  } catch (UnrecognizedProfileTableNameException e) {
    throw e;
  } catch (SLEEException e) {
    throw new ManagementException(e.getMessage(), e);
  } catch (Exception e) {
    throw new ManagementException(e.getMessage(), e);
  } finally {
    try {
      sleeTransactionManagement.requireTransactionEnd(transactionStarted, false);
    } catch (Exception e) {
      throw new ManagementException(e.getMessage(), e);
    }
  }
}