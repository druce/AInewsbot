
""" pydantic classes used to define structured outputs from LLM"""


from typing import List, TypeVar

from pydantic import BaseModel, Field

# so we can pass types to functions
T = TypeVar('T', bound=BaseModel)


class Story(BaseModel):
    """Story class for structured output filtering"""
    id: int = Field(description="The id of the story")
    isAI: bool = Field(description="true if the story is about AI, else false")


class Stories(BaseModel):
    """Stories class for structured output filtering of a list of Story"""
    items: List[Story] = Field(description="List of Story")

# alternatively could only define Story, but no Stories
# then we pass output_class to filter_page where we then define a class as a list of output_class
# class ItemList(RootModel[list[Story or int]]):
#     """List of Story / story indexes"""
#     pass
# and use ItemList as the output class for .with_structured_output
# a little cleaner, no pairs of classes or items field, but not worth changing now


class TopicSpec(BaseModel):
    """TopicSpec class for structured output of story topics"""
    id: int = Field(description="The id of the story")
    extracted_topics: List[str] = Field(
        description="List of topics covered in the story")


class TopicSpecList(BaseModel):
    """List of TopicSpec class for structured output"""
    items: List[TopicSpec] = Field(description="List of TopicSpec")


class SingleTopicSpec(BaseModel):
    """SingleTopicSpec class for structured output of story topic"""
    id: int = Field(description="The id of the story")
    topic: str = Field(
        description="The topic covered in the story")


class CanonicalTopicSpec(BaseModel):
    """CanonicalTopicSpec class for structured output of canonical topics"""
    id: int = Field(description="The id of the story")
    relevant: bool = Field(
        description="True if the story is about the topic else false")


class CanonicalTopicSpecList(BaseModel):
    """List of CanonicalTopicSpec for structured output"""
    items: List[CanonicalTopicSpec] = Field(
        description="List of CanonicalTopicSpec")


class TopicHeadline(BaseModel):
    """Topic headline of a group of stories for structured output"""
    topic_title: str = Field(description="The title for the headline group")


class TopicCategoryList(BaseModel):
    """List of topics for structured output filtering"""
    items: List[str] = Field(description="List of topics")


class Site(BaseModel):
    """Site class for structured output filtering"""
    url: str = Field(description="The URL of the site")
    site_name: str = Field(description="The name of the site")


class Sites(BaseModel):
    """List of Site class for structured output filtering"""
    items: List[Site] = Field(description="List of Site")


class StoryRating(BaseModel):
    """StoryRating class for generic structured output rating"""
    id: int = Field(description="The id of the story")
    rating: int = Field(description="An integer rating of the story")


class StoryRatings(BaseModel):
    """StoryRatings class for structured output filtering of a list of Story"""
    items: List[StoryRating] = Field(description="List of StoryRating")


class NewsArticle(BaseModel):
    """NewsArticle class for structured output filtering"""
    src: str = Field(description="The source of the story")
    url: str = Field(description="The URL of the story")
    summary: str = Field(description="A summary of the story")

    def __str__(self):
        return f"- {self.summary} - [{self.src}]({self.url})"


class StoryOrder(BaseModel):
    """StoryOrder class for generic structured output rating"""
    id: int = Field(description="The id of the story")


class StoryOrderList(BaseModel):
    """List of StoryOrder for structured output"""
    items: List[StoryOrder] = Field(
        description="List of StoryOrder")


class Section(BaseModel):
    """Section class for structured output filtering"""
    section_title: str = Field(description="The title of the section")
    news_items: List[NewsArticle]

    def __str__(self):
        retval = f"## {self.section_title}\n\n"
        retval += "\n".join(
            [str(news_item) for news_item in self.news_items]
        )
        return retval


class Newsletter(BaseModel):
    """Newsletter class for structured output filtering"""
    section_items: List[Section]

    def __str__(self):
        return "\n\n".join(str(section) for section in self.section_items)
