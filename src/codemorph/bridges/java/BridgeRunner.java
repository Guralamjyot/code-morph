package codemorph.bridge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 * Java bridge runner for subprocess-based cross-language calls.
 *
 * <p>This class is invoked from Python as:
 * java -cp ... codemorph.bridge.BridgeRunner com.example.MyClass myMethod
 *
 * <p>It:
 * 1. Reads JSON from stdin
 * 2. Deserializes to Java objects
 * 3. Calls the specified method via reflection
 * 4. Serializes the result to JSON
 * 5. Writes to stdout
 *
 * <p>Used by Python code to call Java functions for I/O verification.
 */
public class BridgeRunner {

    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(
                    "{\"error\": \"Usage: BridgeRunner <className> <methodName>\"}");
            System.exit(1);
        }

        String className = args[0];
        String methodName = args[1];

        try {
            // Read JSON input from stdin
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            StringBuilder inputBuilder = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                inputBuilder.append(line);
            }

            String inputJson = inputBuilder.toString();

            if (inputJson.trim().isEmpty()) {
                System.err.println("{\"error\": \"No input received from stdin\"}");
                System.exit(1);
            }

            // Parse input JSON
            JsonNode inputNode = mapper.readTree(inputJson);

            // Load the target class
            Class<?> targetClass = Class.forName(className);

            // Find the method (simplified - assumes unique method names)
            Method targetMethod = null;
            for (Method method : targetClass.getDeclaredMethods()) {
                if (method.getName().equals(methodName)) {
                    targetMethod = method;
                    break;
                }
            }

            if (targetMethod == null) {
                System.err.println(
                        String.format(
                                "{\"error\": \"Method '%s' not found in class '%s'\"}",
                                methodName, className));
                System.exit(1);
            }

            // Convert JSON args to method parameters
            // For simplicity, assume the method accepts a Map<String, Object>
            // Or individual parameters that can be extracted from JSON
            Map<String, Object> argsMap = mapper.convertValue(inputNode, Map.class);

            // Call the method
            // This is simplified - real implementation needs type matching
            Object result = targetMethod.invoke(null, argsMap); // Assumes static method

            // Serialize result to JSON
            String outputJson = mapper.writeValueAsString(result);

            // Write to stdout
            System.out.println(outputJson);
            System.exit(0);

        } catch (ClassNotFoundException e) {
            System.err.println(
                    String.format("{\"error\": \"Class not found: %s\"}", className));
            System.exit(1);

        } catch (Exception e) {
            System.err.println(
                    String.format(
                            "{\"error\": \"Execution failed: %s\", \"type\": \"%s\"}",
                            e.getMessage(), e.getClass().getSimpleName()));
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
