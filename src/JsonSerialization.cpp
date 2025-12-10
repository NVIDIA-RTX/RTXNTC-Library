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

#include "JsonSerialization.h"

namespace ntc::json
{

void* NtcJsonAllocator::Malloc(size_t size)
{ 
    if (!size)
        return nullptr;
    
    // Allocate a bit more memory to store the allocation size.
    // The NTC allocator interface takes the size with Free(), so we need to know that.
    void* ptr = m_ntcAllocator->Allocate(size + kHeaderSize);
    *((size_t*)ptr) = size;

    return (uint8_t*)ptr + kHeaderSize;
}

void* NtcJsonAllocator::Realloc(void* originalPtr, size_t originalSize, size_t newSize)
{
    void* newPtr = Malloc(newSize);
    if (originalPtr && newPtr)
        memcpy(newPtr, originalPtr, std::min(originalSize, newSize));
    Free(originalPtr);
    return newPtr;        
}

void NtcJsonAllocator::Free(void *ptr) noexcept
{
    if (!ptr)
        return;

    // Obtain the real allocated block
    ptr = (uint8_t*)ptr - kHeaderSize;
    
    // Read the allocation size from the block
    size_t size = *(size_t*)ptr;
    *(size_t*)ptr = 0;

    m_ntcAllocator->Deallocate(ptr, size + kHeaderSize);
}

bool SerializeContext::SerializeObject(void const* object, AbstractObjectSchema const& schema)
{
    // Remember the current type name to restore it later when we leave this object.
    // Lets us avoid using a separate stack for the names and rely on the call stack instead.
    char const* oldTypeName = currentTypeName;
    currentTypeName = schema.GetTypeName();

    // Start writing the JSON object
    writer.StartObject();

    bool success = true;

    // Go over the fields and serialize each one... unless it's an empty optional.
    for (uint32_t i = 0; i < schema.GetNumFields(); ++i)
    {
        Field const& field = schema.GetFields()[i];
        switch (field.type)
        {
        case Field::Type::Bool: {
            bool value = GetObjectField<bool>(object, field.offset);
            writer.Key(field.name);
            writer.Bool(value);
            break;
        }

        case Field::Type::OptionalBool: {
            auto const& optional = GetObjectField<std::optional<bool>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Bool(*optional);
            }
            break;
        }

        case Field::Type::Int: {
            int32_t value = GetObjectField<int32_t>(object, field.offset);
            writer.Key(field.name);
            writer.Int(value);
            break;
        }

        case Field::Type::ArrayOfInt: {
            auto const& vector = GetObjectField<Vector<int>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (int value : vector)
                {
                    writer.Int(value);
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::OptionalInt: {
            auto const& optional = GetObjectField<std::optional<int32_t>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Int(*optional);
            }
            break;
        }

        case Field::Type::UInt: {
            uint32_t value = GetObjectField<uint32_t>(object, field.offset);
            writer.Key(field.name);
            writer.Uint(value);
            break;
        }

        case Field::Type::ArrayOfUInt: {
            auto const& vector = GetObjectField<Vector<uint32_t>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (uint32_t value : vector)
                {
                    writer.Uint(value);
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::OptionalUInt: {
            auto const& optional = GetObjectField<std::optional<uint32_t>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Uint(*optional);
            }
            break;
        }

        case Field::Type::UInt64: {
            uint64_t value = GetObjectField<uint64_t>(object, field.offset);
            writer.Key(field.name);
            writer.Uint64(value);
            break;
        }

        case Field::Type::ArrayOfUInt64: {
            auto const& vector = GetObjectField<Vector<uint64_t>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (uint64_t value : vector)
                {
                    writer.Uint64(value);
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::OptionalUInt64: {
            auto const& optional = GetObjectField<std::optional<uint64_t>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Uint64(*optional);
            }
            break;
        }
        
        case Field::Type::Float: {
            float value = GetObjectField<float>(object, field.offset);
            writer.Key(field.name);
            writer.Double(value);
            break;
        }

        case Field::Type::ArrayOfFloat: {
            auto const& vector = GetObjectField<Vector<float>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (float value : vector)
                {
                    writer.Double(value);
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::OptionalFloat: {
            auto const& optional = GetObjectField<std::optional<float>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Double(*optional);
            }
            break;
        }

        case Field::Type::Double: {
            double value = GetObjectField<double>(object, field.offset);
            writer.Key(field.name);
            writer.Double(value);
            break;
        }

        case Field::Type::ArrayOfDouble: {
            auto const& vector = GetObjectField<Vector<double>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (double value : vector)
                {
                    writer.Double(value);
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::OptionalDouble: {
            auto const& optional = GetObjectField<std::optional<double>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.Double(*optional);
            }
            break;
        }
        case Field::Type::String: {
            String const& value = GetObjectField<String>(object, field.offset);
            writer.Key(field.name);
            writer.String(value.c_str());
            break;
        }

        case Field::Type::OptionalString: {
            auto const& optional = GetObjectField<std::optional<String>>(object, field.offset);
            if (optional.has_value())
            {
                writer.Key(field.name);
                writer.String(optional->c_str());
            }
            break;
        }

        case Field::Type::ArrayOfString: {
            auto const& vector = GetObjectField<Vector<String>>(object, field.offset);
            if (!vector.empty())
            {
                writer.Key(field.name);
                writer.StartArray();
                for (auto const& s : vector)
                {
                    writer.String(s.c_str());
                }
                writer.EndArray();
            }
            break;
        }

        case Field::Type::Object:
        case Field::Type::OptionalObject:
        case Field::Type::ArrayOfObject: {
            // Let the custom handler process the objects because we can't dynamically cast them to an unknown type
            assert(field.objectHandler);
            success = field.objectHandler->Serialize(*this, object, field);
            break;
        }

        case Field::Type::Enum:
        case Field::Type::OptionalEnum: {
            // Let the custom handler process the enums because enums could have different underlying types
            assert(field.enumHandler);
            success = field.enumHandler->Serialize(*this, object, field);
            break;
        }
        
        default:
            break;
        }

        if (!success)
            break;
    }

    // Close the JSON object
    writer.EndObject();

    // Restore the type name
    currentTypeName = oldTypeName;

    return success;
}

char const* ParseHandler::GetErrorMessage() const
{
    return m_errorMessage;
}

bool ParseHandler::Null()
{
    Context* ctx = Top();

    // Skip unknown fields
    if (ctx && !ctx->currentField)
    {
        return true;
    }

    // No field types support Null at this time
    
    SetUnexpectedTypeMessage(ctx, "null");
    return false;
}

bool ParseHandler::Bool(bool b)
{
    Context* ctx = Top();

    if (!ctx)
    {
        SetUnexpectedTypeMessage(ctx, "bool");
        return false;
    }

    // Skip unknown fields
    if (!ctx->currentField)
    {
        return true;
    }

    Field const& field = *ctx->currentField;
    bool success = false;

    switch(field.type)
    {
    case Field::Type::Bool: {
        bool& value = GetObjectField<bool>(ctx->object, field.offset);
        value = b;
        success = true;
        break;
    }
    case Field::Type::OptionalBool: {
        auto& optional = GetObjectField<std::optional<bool>>(ctx->object, field.offset);
        optional = b;
        success = true;
        break;
    }
    }

    if (success)
    {
        MarkFieldAsPresent(ctx);
    }
    else
    {
        SetUnexpectedTypeMessage(ctx, "bool");
    }

    ctx->currentField = nullptr;

    return success;
}

bool ParseHandler::Int(int i)
{
    return Int64(i);
}

bool ParseHandler::Uint(unsigned i)
{
    return Int64(i);
}

bool ParseHandler::Int64(int64_t i)
{
    Context* ctx = Top();

    if (!ctx)
    {
        SetUnexpectedTypeMessage(ctx, "integer");
        return false;
    }

    // Skip unknown fields
    if (!ctx->currentField)
    {
        return true;
    }

    Field const& field = *ctx->currentField;
    bool success = false;
    bool outOfRange = false;

    switch(field.type)
    {
    case Field::Type::Bool: {
        bool& value = GetObjectField<bool>(ctx->object, field.offset);
        // Only accept 0 and 1 for bool fields
        if (i == 0 || i == 1)
        {
            value = i != 0;
            success = true;
        }
        else
            outOfRange = true;
        break;
    }
    case Field::Type::OptionalBool: {
        auto& optional = GetObjectField<std::optional<bool>>(ctx->object, field.offset);
        // Only accept 0 and 1 for bool fields
        if (i == 0 || i == 1)
        {
            optional = i != 0;
            success = true;
        }
        else
            outOfRange = true;
        break;
    }
    case Field::Type::Int: {
        int32_t& value = GetObjectField<int32_t>(ctx->object, field.offset);
        value = int32_t(i);
        if (int64_t(value) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::ArrayOfInt: {
        if (int64_t(int32_t(i)) == i)
        {
            auto& vector = GetObjectField<ntc::Vector<int32_t>>(ctx->object, ctx->currentField->offset);
            vector.push_back(int32_t(i));
            success = true;
        }
        else
            outOfRange = true;
        break;
    }
    case Field::Type::OptionalInt: {
        auto& optional = GetObjectField<std::optional<int32_t>>(ctx->object, field.offset);
        optional = int32_t(i);
        if (int64_t(*optional) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::UInt: {
        uint32_t& value = GetObjectField<uint32_t>(ctx->object, field.offset);
        value = uint32_t(i);
        if (int64_t(value) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::ArrayOfUInt: {
        if (int64_t(uint32_t(i)) == i)
        {
            auto& vector = GetObjectField<ntc::Vector<uint32_t>>(ctx->object, ctx->currentField->offset);
            vector.push_back(uint32_t(i));
            success = true;
        }
        else
            outOfRange = true;
        break;
    }
    case Field::Type::OptionalUInt: {
        auto& optional = GetObjectField<std::optional<uint32_t>>(ctx->object, field.offset);
        optional = uint32_t(i);
        if (int64_t(*optional) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::UInt64: {
        uint64_t& value = GetObjectField<uint64_t>(ctx->object, field.offset);
        value = i;
        if (int64_t(value) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::ArrayOfUInt64: {
        if (i >= 0)
        {
            auto& vector = GetObjectField<ntc::Vector<uint64_t>>(ctx->object, ctx->currentField->offset);
            vector.push_back(uint64_t(i));
            success = true;
        }
        else
            outOfRange = true;
        break;
    }
    case Field::Type::OptionalUInt64: {
        auto& optional = GetObjectField<std::optional<uint64_t>>(ctx->object, field.offset);
        optional = i;
        if (int64_t(*optional) == i)
            success = true;
        else
            outOfRange = true;
        break;
    }
    case Field::Type::Float: {
        float& value = GetObjectField<float>(ctx->object, field.offset);
        value = float(i);
        success = true;
        break;
    }
    case Field::Type::ArrayOfFloat: {
        auto& vector = GetObjectField<ntc::Vector<float>>(ctx->object, ctx->currentField->offset);
        vector.push_back(float(i));
        success = true;
        break;
    }
    case Field::Type::OptionalFloat: {
        auto& optional = GetObjectField<std::optional<float>>(ctx->object, field.offset);
        optional = float(i);
        success = true;
        break;
    }
    case Field::Type::Double: {
        double& value = GetObjectField<double>(ctx->object, field.offset);
        value = double(i);
        success = true;
        break;
    }
    case Field::Type::ArrayOfDouble: {
        auto& vector = GetObjectField<ntc::Vector<double>>(ctx->object, ctx->currentField->offset);
        vector.push_back(double(i));
        success = true;
        break;
    }
    case Field::Type::OptionalDouble: {
        auto& optional = GetObjectField<std::optional<double>>(ctx->object, field.offset);
        optional = double(i);
        success = true;
        break;
    }
    case Field::Type::Enum:
    case Field::Type::OptionalEnum: {
        assert(field.enumHandler);
        success = field.enumHandler->Deserialize(ctx->object, field, nullptr, int(i));
        break;
    }
    }

    if (success)
    {
        if (field.validateFunc)
        {
            if (!field.validateFunc(ctx->object, field, m_errorMessage, sizeof m_errorMessage))
            {
                success = false;
            }
        }

        if (success)
        {
            MarkFieldAsPresent(ctx);
        }
    }
    else
    {
        if (outOfRange)
            snprintf(m_errorMessage, sizeof m_errorMessage, "Value %" PRId64 " is out of range for %s field '%s.%s'",
                i, GetExpectedType(ctx), ctx->schema->GetTypeName(), ctx->currentField->name);
        else
            SetUnexpectedTypeMessage(ctx, "integer");
    }

    if (!ctx->insideArray)
        ctx->currentField = nullptr;

    return success;
}

bool ParseHandler::Uint64(uint64_t i)
{
    return Int64(i);
}

bool ParseHandler::Double(double d)
{
    Context* ctx = Top();

    if (!ctx)
    {
        SetUnexpectedTypeMessage(ctx, "double");
        return false;
    }

    // Skip unknown fields
    if (!ctx->currentField)
    {
        return true;
    }

    Field const& field = *ctx->currentField;
    bool success = false;

    switch(field.type)
    {
    case Field::Type::Float: {
        float& value = GetObjectField<float>(ctx->object, field.offset);
        value = float(d);
        success = true;
        break;
    }
    case Field::Type::ArrayOfFloat: {
        auto& vector = GetObjectField<ntc::Vector<float>>(ctx->object, ctx->currentField->offset);
        vector.push_back(float(d));
        success = true;
        break;
    }
    case Field::Type::OptionalFloat: {
        auto& optional = GetObjectField<std::optional<float>>(ctx->object, field.offset);
        optional = float(d);
        success = true;
        break;
    }
    case Field::Type::Double: {
        double& value = GetObjectField<double>(ctx->object, field.offset);
        value = d;
        success = true;
        break;
    }
    case Field::Type::ArrayOfDouble: {
        auto& vector = GetObjectField<ntc::Vector<double>>(ctx->object, ctx->currentField->offset);
        vector.push_back(d);
        success = true;
        break;
    }
    case Field::Type::OptionalDouble: {
        auto& optional = GetObjectField<std::optional<double>>(ctx->object, field.offset);
        optional = d;
        success = true;
        break;
    }
    }

    if (success)
    {
        MarkFieldAsPresent(ctx);
    }
    else
    {
        SetUnexpectedTypeMessage(ctx, "double");
    }

    if (!ctx->insideArray)
        ctx->currentField = nullptr;

    return success;
}

bool ParseHandler::RawNumber(const char* str, rapidjson::SizeType length, bool copy)
{
    // RawNumber should not be called by the Reader in our context, needs kParseNumbersAsStringsFlag
    assert(false);
    return false;
}

bool ParseHandler::String(const char* str, rapidjson::SizeType length, bool copy)
{
    Context* ctx = Top();

    if (!ctx)
    {
        SetUnexpectedTypeMessage(ctx, "string");
        return false;
    }

    // Skip unknown fields
    if (!ctx->currentField)
    {
        return true;
    }

    Field const& field = *ctx->currentField;
    bool success = false;
    bool unknownValue = false;

    switch(field.type)
    {
    case Field::Type::String: {
        ntc::String& value = GetObjectField<ntc::String>(ctx->object, ctx->currentField->offset);
        value.assign(str);
        success = true;
        break;
    }
    case Field::Type::OptionalString: {
        auto& value = GetObjectField<std::optional<ntc::String>>(ctx->object, ctx->currentField->offset);
        value = ntc::String(str, m_allocator);
        success = true;
        break;
    }
    case Field::Type::ArrayOfString: {
        auto& vector = GetObjectField<ntc::Vector<ntc::String>>(ctx->object, ctx->currentField->offset);
        vector.emplace_back(m_allocator).assign(str);
        success = true;
        break;
    }
    case Field::Type::Enum:
    case Field::Type::OptionalEnum: {
        assert(field.enumHandler);
        success = field.enumHandler->Deserialize(ctx->object, field, str, 0);
        if (!success)
            unknownValue = true;
        break;
    }
    }

    if (success)
    {
        MarkFieldAsPresent(ctx);
    }
    else
    {
        if (unknownValue)
        {
            snprintf(m_errorMessage, sizeof m_errorMessage, "Unknown value '%s' specified for '%s.%s'",
                str, ctx->schema->GetTypeName(), field.name);
        }
        else
            SetUnexpectedTypeMessage(ctx, "string");
    }

    if (!ctx->insideArray)
        ctx->currentField = nullptr;

    return success;
}

bool ParseHandler::StartObject()
{
    Context* ctx = Top();

    if (!ctx)
    {
        Context newCtx;

        newCtx.object = m_document;
        newCtx.schema = &m_documentSchema;
        
        assert(newCtx.schema->GetNumFields() <= 64); // using a uint64_t to track field presence

        m_stack.push_back(newCtx);
        return true;
    }

    if (!ctx->currentField)
    {
        Context newCtx;
        m_stack.push_back(newCtx);
        return true;
    }

    Field const& field = *ctx->currentField;

    if (field.type == Field::Type::Object ||
        field.type == Field::Type::OptionalObject ||
        field.type == Field::Type::ArrayOfObject)
    {
        Context newCtx;

        AbstractObjectHandler const* objectHandler = field.objectHandler;
        assert(objectHandler);

        newCtx.object = objectHandler->BeginParsing(ctx->object, *ctx->currentField, m_allocator);
        
        newCtx.schema = &objectHandler->GetSchema();

        assert(newCtx.schema->GetNumFields() <= 64); // using a uint64_t to track field presence

        m_stack.push_back(newCtx);

        return true;
    }

    SetUnexpectedTypeMessage(ctx, "object");
    return false;
}

bool ParseHandler::Key(const char* str, rapidjson::SizeType length, bool copy)
{
    Context* ctx = Top();
    if (!ctx)
    {
        SetUnexpectedTypeMessage(ctx, "key"); // shouldn't happen, that would be invalid json grammar
        return false;
    }

    // Find a field with a mathcing name in the current object's schema
    ctx->currentField = nullptr;
    if (ctx->schema)
    {
        Field const* fields = ctx->schema->GetFields();
        for (uint32_t i = 0; i < ctx->schema->GetNumFields(); ++i)
        {
            if (strcmp(fields[i].name, str) == 0)
            {
                ctx->currentField = &fields[i];
                ctx->currentFieldIndex = i;
                break;
            }
        }
    }

    // If no such field is found in the schema, process an unknown field by keeping currentField = nullptr

    return true;
}

bool ParseHandler::EndObject(rapidjson::SizeType memberCount)
{
    if (m_stack.empty())
    {
        snprintf(m_errorMessage, sizeof m_errorMessage, "End-of-object encountered outside of an object"); // shouldn't happen
        return false;
    }

    Context* ctx = Top();
    if (ctx->schema)
    {
        // Validate that all fields marked as required by the schema have been provided in the JSON object

        Field const* fields = ctx->schema->GetFields();
        for (uint32_t i = 0; i < ctx->schema->GetNumFields(); ++i)
        {
            Field const& field = fields[i];
            if (!field.required)
                continue;

            bool present = (ctx->presentFieldMask & (1ull << i)) != 0;

            if (!present)
            {
                snprintf(m_errorMessage, sizeof m_errorMessage, "Missing value for required field '%s.%s'",
                    ctx->schema->GetTypeName(), field.name);
                return false;
            }
        }
    }

    // Go up in the object stack
    m_stack.pop_back();

    ctx = Top();
    if (ctx && !ctx->insideArray)
    {
        // If this object was a field in another object, mark that field as present
        MarkFieldAsPresent(ctx);
        ctx->currentField = nullptr;
    }

    return true;
}

bool ParseHandler::StartArray()
{
    Context* ctx = Top();

    // Validate that we're expecting an array or processing an unknown field
    if (ctx && (!ctx->currentField || 
                ctx->currentField->type == Field::Type::ArrayOfInt ||
                ctx->currentField->type == Field::Type::ArrayOfUInt ||
                ctx->currentField->type == Field::Type::ArrayOfUInt64 ||
                ctx->currentField->type == Field::Type::ArrayOfFloat ||
                ctx->currentField->type == Field::Type::ArrayOfDouble ||
                ctx->currentField->type == Field::Type::ArrayOfString ||
                ctx->currentField->type == Field::Type::ArrayOfObject))
    {
        ctx->insideArray = true;
        return true;
    }

    SetUnexpectedTypeMessage(ctx, "array");
    return false;
}

bool ParseHandler::EndArray(rapidjson::SizeType elementCount)
{
    Context* ctx = Top();

    if (ctx && ctx->insideArray)
    {
        // If this array was a field in an object, mark that field as present
        MarkFieldAsPresent(ctx);
        ctx->insideArray = false;
        ctx->currentField = nullptr;
        return true;
    }

    snprintf(m_errorMessage, sizeof m_errorMessage, "End-of-array encountered outside of an array"); // shouldn't happen
    return false;
}

ParseHandler::Context* ParseHandler::Top()
{
    return m_stack.empty() ? nullptr : &m_stack[m_stack.size() - 1];
}

char const* ParseHandler::GetExpectedType(Context* ctx)
{
    if (!ctx)
        return "object";
    
    if (!ctx->currentField)
        return "<unknown>";

    switch(ctx->currentField->type)
    {
    case Field::Type::Bool: 
    case Field::Type::OptionalBool:
        return "bool";

    case Field::Type::Int: 
    case Field::Type::OptionalInt:
    case Field::Type::ArrayOfInt:
        return "integer";

    case Field::Type::UInt: 
    case Field::Type::OptionalUInt:
    case Field::Type::ArrayOfUInt:
        return "uint";

    case Field::Type::UInt64:
    case Field::Type::OptionalUInt64:
    case Field::Type::ArrayOfUInt64:
        return "uint64";

    case Field::Type::Float:
    case Field::Type::OptionalFloat:
    case Field::Type::ArrayOfFloat:
        return "float";

    case Field::Type::Double:
    case Field::Type::OptionalDouble:
    case Field::Type::ArrayOfDouble:
        return "double";

    case Field::Type::String:
    case Field::Type::OptionalString:
    case Field::Type::ArrayOfString:
        return "string";

    case Field::Type::Object:
    case Field::Type::OptionalObject:
        return "object";
        
    case Field::Type::ArrayOfObject:
        return "array";

    case Field::Type::Enum:
    case Field::Type::OptionalEnum:
        return "enum";

    default:
        return "<invalid-type>";
    }
}

void ParseHandler::SetUnexpectedTypeMessage(Context* ctx, char const* actualType)
{
    if (ctx && ctx->schema && ctx->currentField)
    {
        snprintf(m_errorMessage, sizeof m_errorMessage, "Expected %s for '%s.%s', got %s",
            GetExpectedType(ctx), ctx->schema->GetTypeName(), ctx->currentField->name, actualType);
    }
    else
    {
        snprintf(m_errorMessage, sizeof m_errorMessage, "Expected %s, got %s",
            GetExpectedType(ctx), actualType);
    }
}

void ParseHandler::MarkFieldAsPresent(Context* ctx)
{
    ctx->presentFieldMask |= (1ull << ctx->currentFieldIndex);
}

bool SerializeAbstractDocument(void const* document, AbstractObjectSchema const& schema,
    IAllocator* ntcAllocator, String& outString, String& outError, size_t expectedOutputSize)
{
    NtcJsonAllocator allocator(ntcAllocator);
    JsonStringBuffer stringBuffer(&allocator, expectedOutputSize);
    JsonWriter writer(stringBuffer, &allocator);
    SerializeContext context(writer);

    bool success = context.SerializeObject(document, schema);

    outError.assign(context.errorMessage);
    outString.assign(stringBuffer.GetString());

    return success;
}

bool ParseAbstractDocument(void* outDocument, AbstractObjectSchema const& schema,
    IAllocator* ntcAllocator, char* input, String& outErrorMessage)
{
    NtcJsonAllocator allocator(ntcAllocator);
    JsonReader reader(&allocator);
    rapidjson::InsituStringStream inputStream(input);

    ParseHandler handler(ntcAllocator, outDocument, schema);

    bool success = reader.Parse(inputStream, handler);
    
    if (!success)
    {
        // Get the error message from the parser or the handler
        rapidjson::ParseErrorCode errorCode = reader.GetParseErrorCode();
        char const* errorMessage = (errorCode == rapidjson::kParseErrorTermination)
            ? handler.GetErrorMessage()
            : rapidjson::GetParseError_En(errorCode);

        // Extract a snippet of the input from the error location.
        // Can't go back for extra context because we're using in-situ parsing, and the input before that location
        // is likely corrupted.
        size_t errorOffset = reader.GetErrorOffset();
        char const* errorLoc = input + errorOffset;
        char errorBuf[32];
        size_t maxErrorLen = sizeof errorBuf - 4;
        strncpy(errorBuf, errorLoc, maxErrorLen);
        errorBuf[maxErrorLen] = 0;
        if (strlen(errorLoc) > maxErrorLen)
            strcat(errorBuf, "...");

        // Format the final error message directly into 'outErrorMessage'
        outErrorMessage.resize(512);
        int len = snprintf(outErrorMessage.data(), outErrorMessage.size(), "Error at character %zu (%s): %s",
            errorOffset, errorBuf, errorMessage);
        outErrorMessage.resize(len);
    }
    else
    {
        outErrorMessage.clear();
    }

    return success;
}

}