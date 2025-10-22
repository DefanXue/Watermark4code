@SuppressWarnings("Duplicates")
    public static void main(String[] args) {
        // init logging
        PropertyConfigurator.configure(
                CasaItWritingExample.class.getResource(PACKAGE + "/log4j.properties"));

        // create a Container object with some example data
        // this object corresponds to the <container> root element in XML
        Container container = FACTORY.createContainer();
        container.setRealestateitems(FACTORY.createContainerRealestateitems());

        // append some example objects to the Container object
        // Using a loop for repetitive additions
        for (int i = 0; i < 3; i++) {
            container.getRealestateitems().getRealestate().add(createRealestate());
        }

        // convert the Container object into a XML document
        CasaItDocument doc = null;
        try {
            doc = CasaItDocument.newDocument(container);
        } catch (Exception ex) {
            LOGGER.error("Can't create XML document!");
            LOGGER.error("> " + ex.getLocalizedMessage(), ex);
            System.exit(1);
        }

        // write XML document into a java.io.File
        try {
            File tempFile = File.createTempFile("output-", ".xml");
            write(doc, tempFile);
            // Optionally, log the path of the created temp file for debugging/verification
            // LOGGER.info("XML written to temporary file: " + tempFile.getAbsolutePath());
        } catch (IOException ex) {
            LOGGER.error("Can't create temporary file!");
            LOGGER.error("> " + ex.getLocalizedMessage(), ex);
            System.exit(1);
        }

        // write XML document into a java.io.OutputStream
        write(doc, new NullOutputStream());

        // write XML document into a java.io.Writer
        write(doc, new NullWriter());

        // write XML document into a string and send it to the console
        writeToConsole(doc);
    }