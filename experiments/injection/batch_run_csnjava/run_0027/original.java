public void addComponent(DeployableComponent dc) {
    if (logger.isTraceEnabled())
      logger.trace("Adding Component " + dc.getComponentKey());

    // Add the component ..
    components.add(dc);

    // .. the key ..
    componentIDs.add(dc.getComponentKey());

    // .. the dependencies ..
    dependencies.addAll(dc.getDependencies());

    // .. the install actions to be taken ..
    installActions.addAll(dc.getInstallActions());

    // .. post-install actions (if any) ..
    Collection<ManagementAction> postInstallActionsStrings = postInstallActions
    .remove(dc.getComponentKey());

    if (postInstallActionsStrings != null
        && !postInstallActionsStrings.isEmpty()) {
      installActions.addAll(postInstallActionsStrings);
    }
    
    // .. pre-uninstall actions (if any) ..
    Collection<ManagementAction> preUninstallActionsStrings = preUninstallActions
    .remove(dc.getComponentKey());

    if (preUninstallActionsStrings != null)
      uninstallActions.addAll(preUninstallActionsStrings);

    // .. and finally the uninstall actions to the DU.
    uninstallActions.addAll(dc.getUninstallActions());
  }