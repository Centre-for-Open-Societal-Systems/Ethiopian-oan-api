from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, field_serializer
import hashlib
import time
import random

class EventType(str, Enum):
    OE_START = "OE_START"
    OE_END = "OE_END"
    OE_ITEM_RESPONSE = "OE_ITEM_RESPONSE"
    OE_INTERACT = "OE_INTERACT"
    OE_ASSESS = "OE_ASSESS"
    OE_LEVEL_SET = "OE_LEVEL_SET"
    OE_MEDIA = "OE_MEDIA"
    OE_TRANSLATION = "OE_TRANSLATION"
    OE_MODERATION = "OE_MODERATION"

class PData(BaseModel):
    id: str
    ver: str
    pid: Optional[str] = None

class GData(BaseModel):
    id: str
    ver: str

class Target(BaseModel):
    id: str
    ver: str
    type: str
    parent: Optional[Dict[str, str]] = None
    questionsDetails: Optional[Dict[str, Any]] = None
    ttsResponseDetails: Optional[Dict[str, Any]] = None
    asrResponseDetails: Optional[Dict[str, Any]] = None

class BaseEventData(BaseModel):
    pass

class ItemResponseEks(BaseEventData):
    target: Target
    qid: str
    type: str
    state: str
    errorDetails: Optional[Dict[str, Any]] = None

class EndEventEks(BaseEventData):
    progress: int
    stageid: str = ""
    length: float

class StartEventEks(BaseEventData):
    pass

class MediaEventEks(BaseEventData):
    type: str
    media_type: str
    media_id: str
    session_id: str
    storage: Dict[str, str]
    duration: Optional[float] = None

class TranslationEventEks(BaseEventData):
    target: Target
    type: str

class QueryTranslationEventEks(BaseEventData):
    target: Target
    type: str

class ModerationEventEks(BaseEventData):
    target: Target
    type: str

class ApiCallEventEks(BaseEventData):
    target: Target
    type: str

class EData(BaseModel):
    eks: Union[Dict[str, Any], BaseEventData]

class TelemetryEvent(BaseModel):
    eid: Union[EventType, str]
    ver: str = "2.2"
    mid: str = ""
    ets: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    channel: str
    pdata: PData
    gdata: GData
    cdata: List[Dict[str, Any]] = Field(default_factory=list)
    uid: str
    sid: str = ""
    did: str
    edata: EData
    etags: Dict[str, List[Any]] = Field(default_factory=lambda: {"partner": []})

    @field_validator("mid", mode="before")
    def generate_mid_if_empty(cls, values):
        if not values.get("mid"):
            random_str = f"{time.time()}{random.random()}"
            values["mid"] = f"OE_{hashlib.md5(random_str.encode()).hexdigest()}"
        return values

    @field_serializer("eid")
    def serialize_eid(self, eid: Union[EventType, str]) -> str:
        if isinstance(eid, EventType):
            return eid.value
        return eid

class TelemetryRequest(BaseModel):
    id: str = "ekstep.telemetry"
    ver: str = "2.2"
    ets: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    events: List[TelemetryEvent]

def create_event(
    event_type: EventType,
    event_data: Union[Dict[str, Any], BaseEventData],
    uid: str,
    channel: str = "OAN",
    did: str = "default-email",
    sid: str = "",
    pdata_id: str = "OAN",
    pdata_ver: str = "v1.0",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    event = TelemetryEvent(
        eid=event_type,
        ets=timestamp if timestamp is not None else int(datetime.now().timestamp() * 1000),
        channel=channel,
        pdata=PData(id=pdata_id, ver=pdata_ver),
        gdata=GData(id=gdata_id, ver=gdata_ver),
        uid=uid,
        sid=sid,
        did=did,
        edata=EData(eks=event_data)
    )
    event.eid = event.eid.value
    event.edata.eks = event.edata.eks.model_dump(exclude_none=True)
    return event

def create_question_event(
    question_text: str,
    answer_text: str,
    session_id: str,
    uid: str = "system",
    question_source: str = "chat",
    channel: str = "OAN",
    did: str = "system",
    pdata_id: str = "OAN",
    pdata_ver: str = "v1.0",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    target = Target(
        id="oan_chat",
        ver="v1.0",
        type="Question",
        parent={"id": "oan", "type": "chat_service"},
        questionsDetails={
            "questionText": question_text,
            "answerText": {"answer": answer_text},
            "questionSource": question_source,
            "groupDetails": {}
        }
    )
    return create_event(
        event_type=EventType.OE_ITEM_RESPONSE,
        event_data=ItemResponseEks(
            target=target,
            qid=f"chat_{session_id}",
            type="CHAT_QUERY",
            state=""
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp,
    )

def create_error_event(
    error_text: str,
    session_id: str,
    uid: str = "system",
    channel: str = "OAN",
    did: str = "system",
    pdata_id: str = "OAN",
    pdata_ver: str = "v1.0",
    gdata_id: str = "content_id",
    gdata_ver: str = "content_ver",
    timestamp: Optional[int] = None,
) -> TelemetryEvent:
    target = Target(
        id="oan_chat",
        ver="v1.0",
        type="Error",
        parent={"id": "oan", "type": "chat_service"},
    )
    return create_event(
        event_type=EventType.OE_ITEM_RESPONSE,
        event_data=ItemResponseEks(
            target=target,
            qid=f"err_{session_id}",
            type="CHAT_ERROR",
            state="",
            errorDetails={"errorText": error_text}
        ),
        uid=uid,
        sid=session_id,
        channel=channel,
        did=did,
        pdata_id=pdata_id,
        pdata_ver=pdata_ver,
        gdata_id=gdata_id,
        gdata_ver=gdata_ver,
        timestamp=timestamp,
    )
