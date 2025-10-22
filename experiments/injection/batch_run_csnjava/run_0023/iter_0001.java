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
        String name = config != null ? config.getName() : null;
        boolean matches = false;
        if (name != null) {
            for (String prefix : movedConfigs) {
                if (name.startsWith(prefix)) {
                    matches = true;
                    break;
                }
            }
        }

        if (matches) {
            configsToRemove.add(config);
            String newName = name != null ? name.replace("conf.", "conf.client.") : name;
            configsToAdd.add(new Config(newName, config.getValue()));
        }
    }

    configsToAdd.add(new Config("conf.client.transferEncoding", RequestEntityProcessing.CHUNKED));

    configs.removeAll(configsToRemove);
    configs.addAll(configsToAdd);
}