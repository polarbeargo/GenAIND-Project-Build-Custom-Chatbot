import kfp
from kfp import dsl


def data_wrangling_op() -> dsl.ContainerOp:
    """Define the data wrangling operation."""
    return dsl.ContainerOp(
        name='Data Wrangling',
        image='catzzzlol/Project-Build-Custom-Chatbot',
        command=['python', 'data_wrangling.py'],
    )


def custom_query_op() -> dsl.ContainerOp:
    """Define the custom query operation."""
    return dsl.ContainerOp(
        name='Custom Query',
        image='catzzzlol/Project-Build-Custom-Chatbot',
        command=['python', 'custom_query.py'],
    )


def compare_prompts_op() -> dsl.ContainerOp:
    """Define the compare prompts operation."""
    return dsl.ContainerOp(
        name='Compare Prompts',
        image='catzzzlol/Project-Build-Custom-Chatbot',
        command=['python', 'compare_prompts.py'],
    )


@dsl.pipeline(
    name='Generative AI Project Pipeline',
    description='A pipeline for the Generative AI project with Comet-ML integration'
)
def generative_ai_pipeline() -> None:
    """Define the main pipeline for the Generative AI project."""
    data_wrangling_op()
    custom_query_op()
    compare_prompts_op()


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        generative_ai_pipeline,
        'generative_ai_pipeline.yaml'
    )
