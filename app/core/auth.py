from fastapi import HTTPException, Request, status

from app.config import settings


def get_internal_api_key_header_name() -> str:
    """
    내부 API Key 헤더 이름을 반환한다.

    기본값:
    x-shs-internal-api-key
    """

    header_name = getattr(
        settings,
        "shs_internal_api_key_header",
        "x-shs-internal-api-key",
    )

    return header_name.strip().lower()


def get_expected_internal_api_key() -> str:
    """
    Worker Node가 기대하는 내부 API Key 값을 반환한다.
    """

    return getattr(
        settings,
        "shs_internal_api_key",
        "change-this-secret",
    )


async def verify_internal_api_key(request: Request) -> bool:
    """
    /api/worker/* 전용 내부 인증 함수.

    Central Server가 Worker Node의 내부 API를 호출할 때
    다음 헤더를 포함해야 한다.

        x-shs-internal-api-key: change-this-secret

    헤더 이름과 키 값은 .env에서 설정 가능하다.

        SHS_INTERNAL_API_KEY_HEADER=x-shs-internal-api-key
        SHS_INTERNAL_API_KEY=change-this-secret

    사용 예:

        from fastapi import Depends

        @router.post("/tasks")
        async def submit_task(
            _auth: bool = Depends(verify_internal_api_key),
        ):
            ...

    /api/local/* 라우터에는 이 인증을 적용하지 않는다.
    """

    header_name = get_internal_api_key_header_name()
    expected_api_key = get_expected_internal_api_key()

    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal API key is not configured.",
        )

    provided_api_key = request.headers.get(header_name)

    if not provided_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing internal API key header: {header_name}",
        )

    if provided_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key.",
        )

    return True