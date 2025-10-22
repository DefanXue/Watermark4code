public static void upgradeToJerseyConfigBean(List<Config> configs) {
    List<Config> configsToAdd = new ArrayList<>();
    List<Config> configsToRemove = new ArrayList<>();
    List<String> movedConfigs = ImmutableList.of(
        "conf.requestTimeoutMillis",
        "conf.numThreads",
        "conf.authType",
        "conf.oauth",
        "conf.basicAuth",
        "conf.useProxy",
        "conf.proxy",
        "conf.sslConfig"
    );
    for (Config config : configs) {
      if (hasPrefixIn(movedConfigs, config.getName())) {
        configsToRemove.add(config);
        configsToAdd.add(new Config(config.getName().replace("conf.", "conf.client."), config.getValue()));
      }
    }

    configsToAdd.add(new Config("conf.client.transferEncoding", RequestEntityProcessing.CHUNKED));

    configs.removeAll(configsToRemove);
    configs.addAll(configsToAdd);
  }