/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "StdTypes.h"
#include <optional>
#include <functional>
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>
#include <rapidjson/error/en.h>

#ifdef _MSC_VER // MSVC doesn't have strcasecmp
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#endif

namespace ntc::json
{

// This flag allows storing and loading unknown enum values (those not declared in the schema) as integers.
// It's not portable between compilers and should be disabled.
constexpr bool g_AllowIntEnums = false;

// Adapter for using ntc::IAllocator as the allocator for RapidJSON
class NtcJsonAllocator
{
public:
    // Default constructor for compatibility with RapidJSON's "new Allocator()" statements.
    // Such an allocator will crash on use, but this is not actually used here.
    NtcJsonAllocator()
        : m_ntcAllocator(nullptr)
    { }

    NtcJsonAllocator(IAllocator* allocator)
        : m_ntcAllocator(allocator)
    { }

    static const bool kNeedFree = true;

    void* Malloc(size_t size);
    void* Realloc(void* originalPtr, size_t originalSize, size_t newSize);
    void Free(void *ptr) noexcept;

private:
    static const size_t kHeaderSize = sizeof(size_t);
    IAllocator* m_ntcAllocator;
};


typedef rapidjson::ASCII<char> InMemoryEncoding;
typedef rapidjson::ASCII<char> StorageEncoding;
typedef rapidjson::GenericStringBuffer<StorageEncoding, NtcJsonAllocator> JsonStringBuffer;
typedef rapidjson::Writer<JsonStringBuffer, InMemoryEncoding, StorageEncoding, NtcJsonAllocator> JsonWriter;
typedef rapidjson::GenericReader<StorageEncoding, InMemoryEncoding, NtcJsonAllocator> JsonReader;

struct Field;

// A base class for storing object schemas and providing them to various consumers in a type-safe way.
// Doesn't really store any schema information, use ObjectSchema<N> for that.
class AbstractObjectSchema
{
public:
    Field const* GetFields() const { return m_pFields; }
    uint32_t GetNumFields() const { return m_numFields; }
    char const* GetTypeName() const { return m_typeName; }

protected:
    AbstractObjectSchema(char const* typeName, Field const* pFields, uint32_t numFields)
        : m_typeName(typeName)
        , m_pFields(pFields)
        , m_numFields(numFields)
    { }

private:
    char const* m_typeName = nullptr;
    Field const* m_pFields = nullptr;
    uint32_t m_numFields = 0;
};

// Internal context object for serialization
class SerializeContext
{
public:
    char errorMessage[256];
    char const* currentTypeName = nullptr;
    JsonWriter& writer;

    SerializeContext(JsonWriter& _writer)
        : writer(_writer)
    {
        errorMessage[0] = 0;
    }

    bool SerializeObject(void const* object, AbstractObjectSchema const& schema);
};

// Locates a filed of type T inside an object -- constant version
template<typename T>
T const& GetObjectField(void const* object, size_t offset)
{
    return *reinterpret_cast<T const*>(reinterpret_cast<uint8_t const*>(object) + offset);
}

// Locates a filed of type T inside an object -- mutable version
template<typename T>
T& GetObjectField(void* object, size_t offset)
{
    return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(object) + offset);
}

// A base class for serialization and parsing handlers of objects with fields.
// Do not use directly, use one of the derived template classes below (ObjectHandler etc.)
class AbstractObjectHandler
{
public:
    AbstractObjectHandler(AbstractObjectSchema const& schema)
        : m_schema(schema)
    { }

    AbstractObjectSchema const& GetSchema() const
    {
        return m_schema;
    }

    virtual ~AbstractObjectHandler() = default;

    // Called by the serializer to store an object of this type.
    virtual bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const = 0;

    // Called by the parser when a JSON object of this type is opened.
    // Returns a pointer to the new object that should be populated from the JSON object.
    // The schema for the new object can be obtained with GetSchema().
    virtual void* BeginParsing(void* parentObject, Field const& parentField, IAllocator* allocator) const = 0;

protected:
    AbstractObjectSchema const& m_schema;
};

// A structure that associates a value (normally from an enum) with a name
template<typename T>
struct EnumValue
{
    T value;
    char const* name;

    EnumValue()
        : value(T(0))
        , name(nullptr)
    { }

    EnumValue(T value, char const* name)
        : value(value)
        , name(name)
    { }
};

// A base class for serialization and parsing handlers of enums.
// Do not use directly, use the EnumHandler template class below.
class AbstractEnumHandler
{
public:
    virtual ~AbstractEnumHandler() = default;
    virtual bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const = 0;
    virtual bool Deserialize(void* parentObject, Field const& parentField, char const* stringValue, int intValue) const = 0;
};


template<typename T> class ObjectHandler;
template<typename T> class OptionalObjectHandler;
template<typename T> class ArrayOfObjectHandler;
template<typename T, int N> class EnumHandler;

typedef bool (*ValidateValueFunc)(void const* parentObject, Field const& parentField,
    char* outErrorMessage, size_t errorMessageSize);

// A structure that describes one field of a serializable object.
// Use the type-safe static functions within to construct Field objects.
struct Field
{
    enum class Type
    {
        Bool,
        Int,
        UInt,
        UInt64,
        Float,
        Double,
        String,
        Object,
        Enum,
        ArrayOfInt,
        ArrayOfUInt,
        ArrayOfUInt64,
        ArrayOfFloat,
        ArrayOfDouble,
        ArrayOfString,
        ArrayOfObject,
        OptionalBool,
        OptionalInt,
        OptionalUInt,
        OptionalUInt64,
        OptionalFloat,
        OptionalDouble,
        OptionalString,
        OptionalObject,
        OptionalEnum
    };

    char const* name = nullptr;
    Type type = Type::Int;
    size_t offset = 0;
    bool required = false;
    AbstractObjectHandler const* objectHandler = nullptr;
    AbstractEnumHandler const* enumHandler = nullptr;
    ValidateValueFunc validateFunc = nullptr;

    // bool b;
    template<typename ParentType>
    static constexpr Field Bool(char const* name, bool ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::Bool;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // std::optional<bool> b;
    template<typename ParentType>
    static constexpr Field OptionalBool(char const* name, std::optional<bool> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalBool;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // int i;
    template<typename ParentType>
    static constexpr Field Int(char const* name, int ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::Int;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // ntc::Vector<int> v;
    template<typename ParentType>
    static constexpr Field ArrayOfInt(char const* name, ntc::Vector<int> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfInt;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // std::optional<int> i;
    template<typename ParentType>
    static constexpr Field OptionalInt(char const* name, std::optional<int> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalInt;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // uint32_t i;
    template<typename ParentType>
    static constexpr Field UInt(char const* name, uint32_t ParentType::* ptr, ValidateValueFunc validateFunc = nullptr)
    {
        Field res;
        res.name = name;
        res.type = Type::UInt;
        res.offset = GetMemberOffset(ptr);
        res.validateFunc = validateFunc;
        res.required = true;
        return res;
    }

    // ntc::Vector<uint32_t> v;
    template<typename ParentType>
    static constexpr Field ArrayOfUInt(char const* name, ntc::Vector<uint32_t> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfUInt;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // std::optional<uint32_t> i;
    template<typename ParentType>
    static constexpr Field OptionalUInt(char const* name, std::optional<uint32_t> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalUInt;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // uint64_t u;
    template<typename ParentType>
    static constexpr Field UInt64(char const* name, uint64_t ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::UInt64;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // ntc::Vector<uint64_t> v;
    template<typename ParentType>
    static constexpr Field ArrayOfUInt64(char const* name, ntc::Vector<uint64_t> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfUInt64;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // std::optional<uint64_t> u;
    template<typename ParentType>
    static constexpr Field OptionalUInt64(char const* name, std::optional<uint64_t> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalUInt64;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // float f;
    template<typename ParentType>
    static constexpr Field Float(char const* name, float ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::Float;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // ntc::Vector<float> v;
    template<typename ParentType>
    static constexpr Field ArrayOfFloat(char const* name, ntc::Vector<float> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfFloat;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // std::optional<float> f;
    template<typename ParentType>
    static constexpr Field OptionalFloat(char const* name, std::optional<float> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalFloat;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // double d;
    template<typename ParentType>
    static constexpr Field Double(char const* name, double ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::Double;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // ntc::Vector<double> v;
    template<typename ParentType>
    static constexpr Field ArrayOfDouble(char const* name, ntc::Vector<double> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfDouble;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // std::optional<double> d;
    template<typename ParentType>
    static constexpr Field OptionalDouble(char const* name, std::optional<double> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalDouble;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // ntc::String s;
    template<typename ParentType>
    static constexpr Field String(char const* name, ntc::String ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::String;
        res.offset = GetMemberOffset(ptr);
        res.required = true;
        return res;
    }

    // std::optional<ntc::String> s;
    template<typename ParentType>
    static constexpr Field OptionalString(char const* name, std::optional<ntc::String> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalString;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // ntc::Vector<ntc::String> v;
    template<typename ParentType>
    static constexpr Field ArrayOfString(char const* name, ntc::Vector<ntc::String> ParentType::* ptr)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfString;
        res.offset = GetMemberOffset(ptr);
        res.required = false;
        return res;
    }

    // ObjectType o;
    template<typename ParentType, typename ObjectType>
    static constexpr Field Object(char const* name, ObjectType ParentType::*ptr,
        ObjectHandler<ObjectType> const& handler)
    {
        Field res;
        res.name = name;
        res.type = Type::Object;
        res.offset = GetMemberOffset(ptr);
        res.objectHandler = &handler;
        res.required = true;
        return res;
    }

    // std::optional<ObjectType> o;
    template<typename ParentType, typename ObjectType>
    static constexpr Field OptionalObject(char const* name, std::optional<ObjectType> ParentType::* ptr,
        OptionalObjectHandler<ObjectType> const& handler)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalObject;
        res.offset = GetMemberOffset(ptr);
        res.objectHandler = &handler;
        res.required = false;
        return res;
    }

    // ntc::Vector<ObjectType> v;
    template<typename ParentType, typename ObjectType>
    static constexpr Field ArrayOfObject(char const* name, Vector<ObjectType> ParentType::* ptr,
        ArrayOfObjectHandler<ObjectType> const& handler)
    {
        Field res;
        res.name = name;
        res.type = Type::ArrayOfObject;
        res.offset = GetMemberOffset(ptr);
        res.objectHandler = &handler;
        res.required = false;
        return res;
    }

    // EnumType e;
    template<typename ParentType, typename EnumType, int N>
    static constexpr Field Enum(char const* name, EnumType ParentType::* ptr,
        EnumHandler<EnumType, N> const& handler)
    {
        Field res;
        res.name = name;
        res.type = Type::Enum;
        res.offset = GetMemberOffset(ptr);
        res.enumHandler = &handler;
        res.required = true;
        return res;
    }

    // std::optional<EnumType> e;
    template<typename ParentType, typename EnumType, int N>
    static constexpr Field OptionalEnum(char const* name, std::optional<EnumType> ParentType::* ptr,
        EnumHandler<EnumType, N> const& handler)
    {
        Field res;
        res.name = name;
        res.type = Type::OptionalEnum;
        res.offset = GetMemberOffset(ptr);
        res.enumHandler = &handler;
        res.required = false;
        return res;
    }

private:
    // Helper functoin to convert a pointer-to-member to offset of that member in the object.
    template<typename ParentType, typename MemberType>
    static constexpr size_t GetMemberOffset(MemberType ParentType::* ptr)
    {
        return size_t(&(((ParentType*)nullptr)->*ptr));
    }
};

// Template class for storing an array of Field objects with a statically known size.
// Use this class to declare static schema objects.
template<uint32_t N>
class ObjectSchema : public AbstractObjectSchema
{
public:
    ObjectSchema(char const* typeName, Field const (&fields)[N])
        : AbstractObjectSchema(typeName, m_fields, N)
    {
        for (uint32_t i = 0; i < N; ++i)
            m_fields[i] = fields[i];
    }

private:
    Field m_fields[N];
};

// A wrapper function to make the compiler automatically deduce the template argument N
template<uint32_t N>
constexpr ObjectSchema<N> MakeObjectSchema(char const* typeName, Field const (&fields)[N])
{
    return ObjectSchema<N>(typeName, fields);
}

// A handler for fields that contain a plain, required object.
// Use with Field::Object(...)
template<typename T>
class ObjectHandler : public AbstractObjectHandler
{
public:
    using AbstractObjectHandler::AbstractObjectHandler;

    bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const override
    {
        auto const& value = GetObjectField<T>(parentObject, parentField.offset);

        context.writer.Key(parentField.name);
        if (!context.SerializeObject(&value, m_schema))
            return false;

        return true;
    }

    void* BeginParsing(void* parentObject, Field const& parentField, IAllocator* allocator) const override
    {
        auto& value = GetObjectField<T>(parentObject, parentField.offset);

        // The object is stored in the field directly
        return &value;
    }
};

// A handler for fields that contain std::optional<SomeObject>
// The object type must have a constructor that takes ntc::IAllocator.
// Use with Field::OptionalObject(...)
template<typename T>
class OptionalObjectHandler : public AbstractObjectHandler
{
public:
    using AbstractObjectHandler::AbstractObjectHandler;

    bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const override
    {
        auto const& optional = GetObjectField<std::optional<T>>(parentObject, parentField.offset);

        if (optional.has_value())
        {
            context.writer.Key(parentField.name);
            if (!context.SerializeObject(&*optional, m_schema))
                return false;
        }

        return true;
    }

    void* BeginParsing(void* parentObject, Field const& parentField, IAllocator* allocator) const override
    {
        auto& optional = GetObjectField<std::optional<T>>(parentObject, parentField.offset);

        // Populate the optional with a new object, return that object
        optional = T(allocator);
        return &*optional;
    }
};

// A handler for fields that contain Vector<SomeObject>
// The object type must have a constructor that takes ntc::IAllocator.
// Use with Field::ArrayOfObject(...)
template<typename T>
class ArrayOfObjectHandler : public AbstractObjectHandler
{
public:
    using AbstractObjectHandler::AbstractObjectHandler;

    bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const override
    {
        auto const& vector = GetObjectField<Vector<T>>(parentObject, parentField.offset);

        if (vector.empty())
            return true;

        bool success = true;
        context.writer.Key(parentField.name);
        context.writer.StartArray();
        for (auto const& item : vector)
        {
            if (!context.SerializeObject(&item, m_schema))
            {
                success = false;
                break;
            }
        }
        context.writer.EndArray();
        return success;
    }

    void* BeginParsing(void* parentObject, Field const& parentField, IAllocator* allocator) const override
    {
        auto& vector = GetObjectField<Vector<T>>(parentObject, parentField.offset);

        // Append a new object to the vector, return that object
        T& item = vector.emplace_back(allocator);
        return &item;
    }
};

// A handler for fields that contain enums.
// Use with Field::Enum(...) or Field::OptionalEnum(...)
template<typename T, int N>
class EnumHandler : public AbstractEnumHandler
{
public:
    EnumHandler(EnumValue<T> const (&values)[N])
    {
        for (int i = 0; i < N; ++i)
            m_values[i] = values[i];
    }

    bool Serialize(SerializeContext& context, void const* parentObject, Field const& parentField) const
    {
        // Extract the value from the object, if the value is present
        T value;
        if (parentField.type == Field::Type::OptionalEnum)
        {
            auto const& optional = GetObjectField<std::optional<T>>(parentObject, parentField.offset);

            if (!optional.has_value())
                return true;

            value = *optional;
        }
        else
        {
            value = GetObjectField<T>(parentObject, parentField.offset);
        }

        context.writer.Key(parentField.name);

        // Find the name for this value
        for (int i = 0; i < N; ++i)
        {
            if (m_values[i].value == value)
            {
                context.writer.String(m_values[i].name);
                return true;
            }
        }

        if (g_AllowIntEnums) // Unknown value - store as int (not portable!)
        {
            context.writer.Int(int(value));
            return true;
        }

        // Value is unknown and storing ints is prohibited - report an error
        snprintf(context.errorMessage, sizeof context.errorMessage, "Unknown value %d used for enum '%s.%s'",
            int(value), context.currentTypeName, parentField.name);
        context.writer.Null(); // Placeholder to avoid asserts in rapidjson
        return false;
    }

    virtual bool Deserialize(void* parentObject, Field const& parentField, char const* stringValue, int intValue) const
    {
        T value;
        bool valid = false;

        if (stringValue)
        {
            // Normal operation - the parser has read a string.
            // Find a value corresponding to that string, case-insensitive because why not.
            for (int i = 0; i < N; ++i)
            {
                if (strcasecmp(m_values[i].name, stringValue) == 0)
                {
                    value = m_values[i].value;
                    valid = true;
                    break;
                }
            }
        }
        else if (g_AllowIntEnums)
        {
            // Fallback operation - the parser has read an integer.
            value = T(intValue);
            valid = true;
        }

        if (!valid)
            return false;

        if (parentField.type == Field::Type::OptionalEnum)
        {
            GetObjectField<std::optional<T>>(parentObject, parentField.offset) = value;
        }
        else
        {
            GetObjectField<T>(parentObject, parentField.offset) = value;
        }

        return true;
    }

private:
    EnumValue<T> m_values[N];
};

// A wrapper function to make the compiler automatically deduce the template argument N
template<typename T, uint32_t N>
constexpr EnumHandler<T, N> MakeEnumSchema(EnumValue<T> const (&values)[N])
{
    return EnumHandler<T, N>(values);
}

// Parse handler to use with the RapidJSON SAX parser.
// Implements the logic that converts the JSON values to objects and fields described by the schema.
class ParseHandler
{
private:
    // Parsing context describing the object currently being processed and the active field in this object.
    // The field is selected when a Key is read from JSON and cleared when the field's value is read.
    // If the field is an object or an array, a sub-context is created and pushed onto the stack.
    struct Context
    {
        // Current object being deserialized.
        void* object = nullptr;

        // Schema for the current object.
        // Null schema happens when skipping an unknown object.
        AbstractObjectSchema const* schema = nullptr; 

        // Field that is being deserialized, set when a Key is received from the parser.
        // Null currentField means we're either not processing any field or are processing/skipping an unknown field.
        Field const* currentField = nullptr;

        // Index of currentField in the object's schema.
        uint32_t currentFieldIndex = 0; 

        // Flag that is set when the parser is reading items or objects inside an array described by currentField.
        bool insideArray = false;

        // Bit mask of fields in the object that have been parsed, used for validation.
        uint64_t presentFieldMask = 0; 
    };

public:
    ParseHandler(IAllocator* allocator, void* document, AbstractObjectSchema const& documentSchema)
        : m_stack(allocator)
        , m_allocator(allocator)
        , m_document(document)
        , m_documentSchema(documentSchema)
    {
        m_errorMessage[0] = 0;
    }

    char const* GetErrorMessage() const;
    
    // The below functions form the parse handler interface required by RapidJSON:

    bool Null();
    bool Bool(bool b);
    bool Int(int i);
    bool Uint(unsigned i);
    bool Int64(int64_t i);
    bool Uint64(uint64_t i);
    bool Double(double d);
    bool RawNumber(const char* str, rapidjson::SizeType length, bool copy);
    bool String(const char* str, rapidjson::SizeType length, bool copy);
    bool StartObject();
    bool Key(const char* str, rapidjson::SizeType length, bool copy);
    bool EndObject(rapidjson::SizeType memberCount);
    bool StartArray();
    bool EndArray(rapidjson::SizeType elementCount);

private:

    Vector<Context> m_stack;
    IAllocator* m_allocator;
    void* m_document;
    AbstractObjectSchema const& m_documentSchema;
    char m_errorMessage[256];

    // Returns the active context, or null if we're outside of the document object
    Context* Top();
    char const* GetExpectedType(Context* ctx);
    void SetUnexpectedTypeMessage(Context* ctx, char const* actualType);
    void MarkFieldAsPresent(Context* ctx);
};

// Serializes a document with the provided schema into a JSON string.
// Returns true if successful, false on error.
// Currently, errors may only happen when an enum fails to serialize due to an unknown value.
bool SerializeAbstractDocument(void const* document, AbstractObjectSchema const& schema,
    IAllocator* ntcAllocator, String& outString, String& outError, size_t expectedOutputSize = 0);

// Parses a JSON string into a document with the provided schema and performs basic validation of the required fields.
// Returns true if successful, false on error.
// Warning: in-situ parsing is used, input will be corrupted!
bool ParseAbstractDocument(void* outDocument, AbstractObjectSchema const& schema,
    IAllocator* ntcAllocator, char* input, String& outErrorMessage);

}