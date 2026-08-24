{#
    dbt default generate_schema_name macro concatenates the target schema
    with the custom schema (e.g. target "staging" + custom "marts" becomes
    "staging_marts"). For this project we want raw/staging/marts to be
    literal, standalone schema names, so we override it to always use the
    custom schema as-is when one is set via `+schema:` in dbt_project.yml.
    See: https://docs.getdbt.com/docs/build/custom-schemas
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}