package codemorph.bridge;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Type compatibility checker for JSON serialization round-trip.
 *
 * <p>Implements the type compatibility strategy from Spec Section 10.1:
 * "Use JSON as the universal data interchange format to verify type alignment
 * without writing custom JNI code for every type."
 *
 * <p>This class attempts to deserialize JSON strings into specific Java types
 * to verify that Python values can be correctly interpreted by Java code.
 */
public class TypeChecker {

    private static final ObjectMapper mapper = new ObjectMapper();

    /**
     * Check if a JSON string can be deserialized to a specific Java type.
     *
     * <p>Called from Python via subprocess bridge to verify type compatibility.
     *
     * @param jsonString The JSON string to deserialize
     * @param typeSignature Java type signature (e.g., "List<Integer>", "Map<String,Double>")
     * @return Map with "compatible" boolean and optional "error" message
     */
    public static Map<String, Object> canDeserialize(String jsonString, String typeSignature) {
        Map<String, Object> result = new HashMap<>();

        try {
            // Parse the type signature and attempt deserialization
            Object deserialized = deserializeByTypeSignature(jsonString, typeSignature);

            // If we got here without exception, it's compatible
            result.put("compatible", true);
            result.put("value", deserialized);

            return result;

        } catch (Exception e) {
            // Deserialization failed - types are incompatible
            result.put("compatible", false);
            result.put("error", e.getMessage());
            result.put("error_type", e.getClass().getSimpleName());

            return result;
        }
    }

    /**
     * Deserialize JSON based on a type signature string.
     *
     * <p>Supports common Java types:
     * - Primitives: int, long, double, boolean, String
     * - Collections: List<T>, Map<K,V>
     * - Custom classes (by fully qualified name)
     *
     * @param jsonString The JSON to deserialize
     * @param typeSignature The target type
     * @return The deserialized object
     * @throws Exception If deserialization fails
     */
    private static Object deserializeByTypeSignature(String jsonString, String typeSignature)
            throws Exception {

        // Remove whitespace from type signature
        typeSignature = typeSignature.replaceAll("\\s+", "");

        // Handle common types
        switch (typeSignature) {
            case "int":
            case "Integer":
                return mapper.readValue(jsonString, Integer.class);

            case "long":
            case "Long":
                return mapper.readValue(jsonString, Long.class);

            case "double":
            case "Double":
                return mapper.readValue(jsonString, Double.class);

            case "float":
            case "Float":
                return mapper.readValue(jsonString, Float.class);

            case "boolean":
            case "Boolean":
                return mapper.readValue(jsonString, Boolean.class);

            case "String":
                return mapper.readValue(jsonString, String.class);

            default:
                // Handle generic types
                if (typeSignature.startsWith("List<")) {
                    return deserializeList(jsonString, typeSignature);
                } else if (typeSignature.startsWith("Map<")) {
                    return deserializeMap(jsonString, typeSignature);
                } else {
                    // Assume it's a fully qualified class name
                    Class<?> targetClass = Class.forName(typeSignature);
                    return mapper.readValue(jsonString, targetClass);
                }
        }
    }

    /**
     * Deserialize a List with generic type parameter.
     *
     * @param jsonString The JSON array
     * @param typeSignature The List type (e.g., "List<Integer>")
     * @return The deserialized list
     * @throws Exception If deserialization fails
     */
    private static Object deserializeList(String jsonString, String typeSignature)
            throws Exception {
        // Extract element type from "List<ElementType>"
        String elementType = extractGenericType(typeSignature, "List");

        // Create JavaType for List<ElementType>
        JavaType elementJavaType = getJavaType(elementType);
        JavaType listType = mapper.getTypeFactory().constructCollectionType(List.class, elementJavaType.getRawClass());

        return mapper.readValue(jsonString, listType);
    }

    /**
     * Deserialize a Map with generic type parameters.
     *
     * @param jsonString The JSON object
     * @param typeSignature The Map type (e.g., "Map<String,Integer>")
     * @return The deserialized map
     * @throws Exception If deserialization fails
     */
    private static Object deserializeMap(String jsonString, String typeSignature) throws Exception {
        // Extract key and value types from "Map<KeyType,ValueType>"
        String genericPart = typeSignature.substring(4, typeSignature.length() - 1);
        String[] types = genericPart.split(",", 2);

        if (types.length != 2) {
            throw new IllegalArgumentException(
                    "Invalid Map type signature: " + typeSignature);
        }

        String keyType = types[0].trim();
        String valueType = types[1].trim();

        // Create JavaType for Map<K,V>
        JavaType keyJavaType = getJavaType(keyType);
        JavaType valueJavaType = getJavaType(valueType);
        JavaType mapType =
                mapper.getTypeFactory()
                        .constructMapType(
                                Map.class, keyJavaType.getRawClass(), valueJavaType.getRawClass());

        return mapper.readValue(jsonString, mapType);
    }

    /**
     * Extract the generic type parameter from a type signature.
     *
     * @param typeSignature Full type signature (e.g., "List<Integer>")
     * @param containerType Container type name (e.g., "List")
     * @return The generic type (e.g., "Integer")
     */
    private static String extractGenericType(String typeSignature, String containerType) {
        int start = containerType.length() + 1; // After "Container<"
        int end = typeSignature.lastIndexOf(">");
        return typeSignature.substring(start, end).trim();
    }

    /**
     * Convert a type string to Jackson JavaType.
     *
     * @param typeStr Type as string (e.g., "Integer", "String")
     * @return Corresponding JavaType
     * @throws ClassNotFoundException If class not found
     */
    private static JavaType getJavaType(String typeStr) throws ClassNotFoundException {
        Class<?> clazz;

        switch (typeStr) {
            case "int":
            case "Integer":
                clazz = Integer.class;
                break;
            case "long":
            case "Long":
                clazz = Long.class;
                break;
            case "double":
            case "Double":
                clazz = Double.class;
                break;
            case "float":
            case "Float":
                clazz = Float.class;
                break;
            case "boolean":
            case "Boolean":
                clazz = Boolean.class;
                break;
            case "String":
                clazz = String.class;
                break;
            default:
                clazz = Class.forName(typeStr);
        }

        return mapper.getTypeFactory().constructType(clazz);
    }

    /**
     * Test method for standalone execution.
     *
     * <p>Usage: java codemorph.bridge.TypeChecker
     */
    public static void main(String[] args) {
        // Test type compatibility
        String[] testCases = {
            "{\"jsonString\": \"42\", \"typeSignature\": \"Integer\"}",
            "{\"jsonString\": \"[1,2,3]\", \"typeSignature\": \"List<Integer>\"}",
            "{\"jsonString\": \"{\\\"a\\\":1,\\\"b\\\":2}\", \"typeSignature\": \"Map<String,Integer>\"}"
        };

        System.out.println("Testing type compatibility checks:");

        for (String testCase : testCases) {
            try {
                Map<String, String> input = mapper.readValue(testCase, new TypeReference<Map<String, String>>() {});
                Map<String, Object> result =
                        canDeserialize(input.get("jsonString"), input.get("typeSignature"));

                System.out.println("Input: " + testCase);
                System.out.println("Result: " + mapper.writeValueAsString(result));
                System.out.println();

            } catch (Exception e) {
                System.err.println("Test failed: " + e.getMessage());
            }
        }
    }
}
