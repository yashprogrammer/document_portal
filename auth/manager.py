import os
from typing import Optional, AsyncGenerator
from uuid import UUID

from fastapi import Depends, Request
from fastapi_users import BaseUserManager
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_async_session
from .models import User


AUTH_SECRET = os.getenv("AUTH_SECRET", "CHANGE_ME_SUPER_SECRET")


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(BaseUserManager[User, UUID]):
    reset_password_token_secret = AUTH_SECRET
    verification_token_secret = AUTH_SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        return

    def parse_id(self, user_id: str) -> UUID:  # required to decode JWT subject
        return UUID(user_id)


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncGenerator["UserManager", None]:
    yield UserManager(user_db)


